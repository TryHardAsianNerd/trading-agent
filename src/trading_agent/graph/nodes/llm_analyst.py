"""Node 4, LLM analyst: the one LLM node, proposing direction only as TradeProposals.

The prompt holds prices, features and held symbol names only: no limits, sizes or
halt state, so the model cannot game numbers it never sees. A buy against trend_up
or a sell with it is discarded here (D-060): the prompt guides, the filter guarantees.
Any LLM failure yields no proposals and an audit row, never a crashed cycle.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session, sessionmaker

from trading_agent.graph.state import AgentState, TradeProposal
from trading_agent.log import get_logger
from trading_agent.portfolio.core import is_core
from trading_agent.storage.repos import DecisionRepository

if TYPE_CHECKING:
    from trading_agent.config import Settings
    from trading_agent.graph.deps import CycleDeps

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You are the market analyst of an automated trading system. From the market \
snapshot in the user message, propose trade actions for this cycle.

Rules:
- Propose at most ONE action per symbol, and only for symbols present in the \
snapshot.
- action is "buy", "sell", or "hold".
- features.trend_up is 1 when, at the last completed daily close, the price \
was above its 50-day average AND the 20-day average was above the 50-day; 0 \
otherwise. It is missing when there are fewer than 50 days of history.
- Only propose "buy" when trend_up is 1; any other buy is discarded. When \
trend_up is 1 and the symbol is not in held, lean toward "buy" unless the \
other data argues against it (e.g. strongly negative news, a sharp one-day \
drop in log_return_1d, RSI far into overbought); say what argued against it \
when you hold instead.
- held lists the snapshot symbols the account holds. When a symbol listed in \
held has trend_up 0, propose "sell", with confidence your probability that \
exiting beats staying in. Never propose "sell" for a symbol not listed in \
held or whose trend_up is 1; any sell whose trend_up is not 0 is discarded. \
Only propose "buy" for a symbol not listed in \
held. For anything else, propose "hold".
- confidence is your honest probability (0.0-1.0) that the action outperforms \
holding over the next few days. Do not inflate it.
- reasoning is 1-2 sentences citing the specific data that drove the call.
- features.news_sentiment_24h, when present, is the average tone of the \
symbol's news headlines over the last 24 hours as scored by a financial \
sentiment model, from -1 (all negative) to +1 (all positive), over \
news_count_24h headlines. A headline tagged with many symbols (a market wrap) \
counts fully for each, though it may not be about this one; a few headlines \
is thin evidence.
- features.crypto_fear_greed, on crypto symbols, is the daily Crypto Fear & \
Greed Index for the whole crypto market: 0 = extreme fear, 100 = extreme \
greed. It describes crowd mood, not any one coin.

You only propose direction. A separate deterministic risk engine — outside \
your control and invisible to you — decides position sizes, stops, and \
whether any proposal executes at all. Never mention sizing or order details.\
"""


class AnalystResponse(BaseModel):
    """Structured-output envelope for the LLM: a list of proposals, nothing else.

    A separate wrapper (not an extension of TradeProposal — that schema is
    locked by tests) so the model returns one well-typed object.
    """

    proposals: list[TradeProposal]

    @field_validator("proposals", mode="before")
    @classmethod
    def _coerce_stringified(cls, v: object) -> object:
        """Unwrap proposals the model returned as a JSON string or a nested envelope.

        Claude can stringify the whole response into the field, which pydantic would
        reject; TradeProposal validation still runs on what comes out.
        """
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict) and "proposals" in v:
            v = v["proposals"]
        return v


def analyze(state: AgentState, *, deps: CycleDeps) -> dict:
    """Run the configured analyst; no proposals (and no LLM spend) under any halt."""
    halted, reason = deps.kill_switch.is_halted()
    if halted:
        log.info("analyst_skipped", cycle_id=state.get("cycle_id"), reason=reason)
        return {"proposals": []}
    proposals = deps.analyst(state)
    log.info(
        "analyst_proposals",
        cycle_id=state.get("cycle_id"),
        proposals=[(p.symbol, p.action, p.confidence) for p in proposals],
    )
    return {"proposals": proposals}


class ClaudeAnalyst:
    """The Claude-backed analyst: state snapshot in, TradeProposals out."""

    def __init__(self, settings: Settings, session_factory: sessionmaker[Session]) -> None:
        """Build the structured-output chain once per process; ValueError without an API key."""
        if not settings.anthropic_api_key:
            raise ValueError("ClaudeAnalyst requires ANTHROPIC_API_KEY")
        self._session_factory = session_factory
        self._consecutive_failures = 0
        # No sampling params — Claude Sonnet 5 400s them; cap max_tokens or spend is unbounded.
        self._chain = ChatAnthropic(
            model_name=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens_to_sample=4096,
            timeout=60.0,
            max_retries=2,
        ).with_structured_output(AnalystResponse)

    def __call__(self, state: AgentState) -> list[TradeProposal]:
        """Propose for the priced swing symbols; never raises.

        Empty when nothing is priced, the holdings are unknown, or the call or its
        parsing fails (audited). ML predictions are not shown: they have no edge (D-032).
        """
        market = state.get("market_data") or {}
        # The core sleeve is deterministic (D-041).
        prices = {s: p for s, p in (market.get("prices") or {}).items() if not is_core(s)}
        if not prices:
            # Info, not a warning: this is every cycle outside the NYSE session.
            log.info("analyst_skipped", reason="no swing symbols priced this cycle")
            return []
        if "held" not in market:
            # Without holdings it would miss real exits.
            log.info("analyst_skipped", reason="holdings unknown")
            return []

        features = {s: f for s, f in (state.get("features") or {}).items() if s in prices}
        snapshot = {
            "prices": {s: str(p) for s, p in sorted(prices.items())},
            "asset_classes": {
                s: c for s, c in (market.get("asset_classes") or {}).items() if s in prices
            },
            "features": features,
            "held": [s for s in market["held"] if s in prices],
        }
        try:
            response = self._chain.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    # sort_keys: identical snapshots must yield byte-identical prompts.
                    HumanMessage(content=json.dumps(snapshot, default=str, sort_keys=True)),
                ]
            )
            proposals = response.proposals
        except Exception as e:  # LLM/API/parse failure: contain, audit, hold
            self._consecutive_failures += 1
            log.exception(
                "analyst_llm_failed",
                cycle_id=state.get("cycle_id"),
                consecutive_failures=self._consecutive_failures,
            )
            if self._consecutive_failures >= 3:
                # A dead key/model config looks like a one-off blip — make persistence loud.
                log.error(
                    "analyst_persistently_failing",
                    consecutive_failures=self._consecutive_failures,
                )
            self._audit_failure(state, type(e).__name__)
            return []
        self._consecutive_failures = 0

        kept = [p for p in proposals if p.symbol in prices]
        if len(kept) < len(proposals):
            # Drop hallucinated symbols here; dying at the gate would muddy the audit.
            dropped = [p.symbol for p in proposals if p.symbol not in prices]
            log.warning("analyst_dropped_unpriced_symbols", symbols=dropped)
        required = {"buy": 1.0, "sell": 0.0}
        against = [
            p
            for p in kept
            if p.action in required
            and (features.get(p.symbol) or {}).get("trend_up") != required[p.action]
        ]
        for p in against:
            log.warning(
                f"analyst_{p.action}_against_trend_dropped",
                cycle_id=state.get("cycle_id"),
                symbol=p.symbol,
                trend_up=(features.get(p.symbol) or {}).get("trend_up"),
                confidence=p.confidence,
                reasoning=p.reasoning,
            )
        return [p for p in kept if p not in against]

    def _audit_failure(self, state: AgentState, error: str) -> None:
        """Write the analyst-failure decisions row; best-effort, never raises."""
        try:
            with self._session_factory() as session:
                DecisionRepository(session).log(
                    cycle_id=state["cycle_id"],
                    ts=datetime.now(UTC),
                    node="llm_analyst",
                    detail={"error": error, "proposals": 0},
                )
                session.commit()
        except Exception:
            log.exception("analyst_failure_audit_write_failed")


def stub_analyst(state: AgentState) -> list[TradeProposal]:
    """The analyst without an ANTHROPIC_API_KEY: a zero-confidence hold per priced symbol."""
    market = state.get("market_data") or {}
    prices = market.get("prices") or {}
    asset_classes = market.get("asset_classes") or {}
    return [
        TradeProposal(
            symbol=symbol,
            asset_class=asset_classes[symbol],
            action="hold",
            confidence=0.0,
            reasoning="stub analyst - no ANTHROPIC_API_KEY configured",
        )
        for symbol in sorted(prices)
    ]
