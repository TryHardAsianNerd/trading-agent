"""ClaudeAnalyst tests: mocked LLM chain, no network, no database.

The boundary under test (docs/RISK_RULES.md structural enforcement): the LLM
sees only the market snapshot, produces only TradeProposals, and any failure
yields an empty list plus an audit row — never a crashed cycle.
"""

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from trading_agent.config import Settings
from trading_agent.graph.deps import make_deps
from trading_agent.graph.nodes.llm_analyst import (
    AnalystResponse,
    ClaudeAnalyst,
    stub_analyst,
)
from trading_agent.graph.state import TradeProposal


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, anthropic_api_key="test-key-not-real", **overrides)


class FakeSession:
    """Captures DecisionRepository writes without a database."""

    def __init__(self, added: list) -> None:
        self.added = added
        self.committed = False

    def add(self, obj) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeSessionFactory:
    def __init__(self) -> None:
        self.added: list = []

    def __call__(self) -> FakeSession:
        return FakeSession(self.added)


class BrokenSessionFactory:
    def __call__(self):
        raise RuntimeError("db down")


class FakeChain:
    """Stands in for the structured-output runnable."""

    def __init__(self, response: AnalystResponse | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.response


def _proposal(symbol: str, action: str = "buy") -> TradeProposal:
    return TradeProposal(
        symbol=symbol, asset_class="stock", action=action, confidence=0.7, reasoning="test"
    )


def _analyst(chain: FakeChain, session_factory=None) -> ClaudeAnalyst:
    analyst = ClaudeAnalyst(_settings(), session_factory or FakeSessionFactory())
    analyst._chain = chain
    return analyst


def _state(*symbols: str, held: tuple[str, ...] = ()) -> dict:
    """Symbols priced at 100, each in an uptrend (trend_up 1); held lists the account's holdings."""
    return {
        "cycle_id": "c-1",
        "market_data": {
            "prices": {s: Decimal("100") for s in symbols},
            "asset_classes": dict.fromkeys(symbols, "stock"),
            "held": sorted(held),
        },
        "features": {s: {"trend_up": 1.0} for s in symbols},
        "predictions": {},
    }


def test_requires_api_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ClaudeAnalyst(Settings(_env_file=None), FakeSessionFactory())


def test_returns_chain_proposals_for_priced_symbols():
    chain = FakeChain(AnalystResponse(proposals=[_proposal("AAPL"), _proposal("MSFT", "hold")]))
    proposals = _analyst(chain)(_state("AAPL", "MSFT"))
    assert [(p.symbol, p.action) for p in proposals] == [("AAPL", "buy"), ("MSFT", "hold")]


def test_hallucinated_symbols_are_dropped():
    """A proposal for a symbol not in the snapshot would only die at the gate
    as INSUFFICIENT_DATA; the analyst drops it up front."""
    chain = FakeChain(AnalystResponse(proposals=[_proposal("AAPL"), _proposal("NOT-REAL")]))
    proposals = _analyst(chain)(_state("AAPL"))
    assert [p.symbol for p in proposals] == ["AAPL"]


def test_duplicate_proposals_pass_through_gate_dedups():
    chain = FakeChain(AnalystResponse(proposals=[_proposal("AAPL"), _proposal("AAPL")]))
    proposals = _analyst(chain)(_state("AAPL"))
    assert len(proposals) == 2  # the gate's DUPLICATE_PROPOSAL handling owns this


def test_llm_failure_returns_empty_and_writes_audit_row():
    factory = FakeSessionFactory()
    chain = FakeChain(error=RuntimeError("api down"))
    proposals = _analyst(chain, factory)(_state("AAPL"))
    assert proposals == []
    (row,) = factory.added
    assert row.node == "llm_analyst"
    assert row.cycle_id == "c-1"
    assert row.detail == {"error": "RuntimeError", "proposals": 0}


def test_schema_violation_is_contained_like_any_llm_failure():
    """Structured output that fails validation (e.g. confidence out of range)
    surfaces as an exception from the chain — contained, never raised."""
    with pytest.raises(Exception) as excinfo:
        TradeProposal(
            symbol="AAPL", asset_class="stock", action="buy", confidence=7.0, reasoning="x"
        )
    factory = FakeSessionFactory()
    proposals = _analyst(FakeChain(error=excinfo.value), factory)(_state("AAPL"))
    assert proposals == []
    assert factory.added[0].detail["error"] == "ValidationError"


def test_audit_write_failure_is_also_contained():
    chain = FakeChain(error=RuntimeError("api down"))
    analyst = ClaudeAnalyst(_settings(), BrokenSessionFactory())
    analyst._chain = chain
    assert analyst(_state("AAPL")) == []  # no exception despite LLM AND db failing


def test_no_prices_skips_the_llm_entirely():
    chain = FakeChain(error=AssertionError("must not be invoked"))
    assert _analyst(chain)({"cycle_id": "c-1", "market_data": {"prices": {}}}) == []
    assert chain.calls == []


def test_prompt_payload_contains_only_the_market_snapshot():
    """Structural enforcement: even if the state carries portfolio/risk data,
    the LLM prompt is built from the four snapshot keys and nothing else —
    no risk limits, no sizing hints, no quantities, no kill-switch state."""
    chain = FakeChain(AnalystResponse(proposals=[]))
    state = _state("AAPL", held=("AAPL",))
    state["approved_orders"] = ["SHOULD-NEVER-REACH-PROMPT"]
    state["rejections"] = [{"reason": "kill_switch_active"}]
    state["predictions"] = {"by_symbol": {"AAPL": {"prob_up": 0.9}}}
    _analyst(chain)(state)

    (messages,) = chain.calls
    human = messages[-1].content
    payload = json.loads(human)
    # No "predictions" (D-032): the backtest found the ML model has no edge.
    assert set(payload) == {"prices", "asset_classes", "features", "held"}
    assert payload["held"] == ["AAPL"]
    full_prompt = messages[0].content + human
    assert "quantity" not in full_prompt
    assert "kill_switch" not in full_prompt
    assert "SHOULD-NEVER-REACH-PROMPT" not in full_prompt
    assert "prob_up" not in full_prompt


def test_response_coerces_stringified_tool_args():
    """A proposals arg holding the ENTIRE response object as a JSON string is
    unwrapped; plain garbage still fails validation."""
    stringified = json.dumps(
        {
            "proposals": [
                {
                    "symbol": "BTC-USD",
                    "asset_class": "crypto",
                    "action": "hold",
                    "confidence": 0.2,
                    "reasoning": "no directional trade.",
                }
            ]
        }
    )
    response = AnalystResponse.model_validate({"proposals": stringified})
    (proposal,) = response.proposals
    assert proposal.symbol == "BTC-USD" and proposal.action == "hold"

    # A bare stringified LIST (without the envelope) also parses.
    bare_list = json.dumps([proposal.model_dump()])
    assert len(AnalystResponse.model_validate({"proposals": bare_list}).proposals) == 1

    # Normal shape still validates; garbage still fails.
    assert AnalystResponse(proposals=[_proposal("AAPL")]).proposals[0].symbol == "AAPL"
    with pytest.raises(ValidationError):
        AnalystResponse.model_validate({"proposals": "not json at all"})


def test_constructor_sends_no_sampling_params_and_caps_max_tokens():
    """Wire-level pin the mocked-chain tests can't see: Claude Sonnet 5
    REJECTS sampling params (temperature/top_p/top_k) with a 400 — one such
    kwarg would silently kill the analyst on every production cycle — and
    without an explicit max_tokens the model profile's 128K default makes
    response spend unbounded."""
    analyst = ClaudeAnalyst(_settings(), FakeSessionFactory())
    model = analyst._chain.first.bound  # the ChatAnthropic under the binding
    assert model.temperature is None
    assert model.top_p is None
    assert model.top_k is None
    assert model.max_tokens == 4096


def test_persistent_failure_is_logged_distinctly():
    """Three consecutive LLM failures must emit analyst_persistently_failing —
    a dead key must not be indistinguishable from a transient blip."""
    factory = FakeSessionFactory()
    analyst = _analyst(FakeChain(error=RuntimeError("dead key")), factory)
    with capture_logs() as logs:
        for _ in range(3):
            assert analyst(_state("AAPL")) == []
    assert analyst._consecutive_failures == 3
    assert any(e["event"] == "analyst_persistently_failing" for e in logs)
    # Success resets the streak.
    analyst._chain = FakeChain(AnalystResponse(proposals=[]))
    analyst(_state("AAPL"))
    assert analyst._consecutive_failures == 0


def test_make_deps_selects_claude_analyst_only_with_key():
    factory = FakeSessionFactory()
    with_key = make_deps(_settings(), factory)
    assert isinstance(with_key.analyst, ClaudeAnalyst)
    without_key = make_deps(Settings(_env_file=None), factory)
    assert without_key.analyst is stub_analyst


def test_buys_outside_an_uptrend_are_discarded_and_the_rest_kept():
    """The prompt steers; this filter guarantees. A symbol with no trend label
    (under 50 bars, e.g. new crypto history) gets no buy either."""
    state = _state("AAPL", "MSFT", "BTC-USD")
    state["features"]["MSFT"] = {"trend_up": 0.0}
    state["features"]["BTC-USD"] = {"sma_20": 95.0}  # no 50-day yet: no label
    chain = FakeChain(
        AnalystResponse(
            proposals=[
                _proposal("AAPL"),
                _proposal("MSFT"),
                _proposal("BTC-USD"),
                _proposal("MSFT", "sell"),
            ]
        )
    )
    proposals = _analyst(chain)(state)
    assert [(p.symbol, p.action) for p in proposals] == [("AAPL", "buy"), ("MSFT", "sell")]


def test_sells_outside_a_downtrend_are_discarded_and_the_rest_kept():
    """A sell stands only when trend_up is 0; no label means no trend exit."""
    state = _state("GOOGL", "META", "MSFT", held=("GOOGL", "META", "MSFT"))
    state["features"]["META"] = {"trend_up": 0.0}
    state["features"]["MSFT"] = {"sma_20": 95.0}  # no 50-day yet: no label
    chain = FakeChain(
        AnalystResponse(
            proposals=[
                _proposal("GOOGL", "sell"),
                _proposal("META", "sell"),
                _proposal("MSFT", "sell"),
            ]
        )
    )
    with capture_logs() as logs:
        proposals = _analyst(chain)(state)
    assert [(p.symbol, p.action) for p in proposals] == [("META", "sell")]
    dropped = [e for e in logs if e["event"] == "analyst_sell_against_trend_dropped"]
    assert [(e["symbol"], e["trend_up"], e["reasoning"]) for e in dropped] == [
        ("GOOGL", 1.0, "test"),
        ("MSFT", None, "test"),
    ]


def test_the_prompt_states_the_trend_rule():
    chain = FakeChain(AnalystResponse(proposals=[]))
    _analyst(chain)(_state("AAPL"))
    assert 'Only propose "buy" when trend_up is 1' in chain.calls[0][0].content


def test_the_analyst_never_sees_a_core_symbol():
    """The core sleeve is deterministic (D-041): not priced, featured or classed in the prompt."""
    chain = FakeChain(AnalystResponse(proposals=[]))
    _analyst(chain)(_state("AAPL", "VOO", "BTC-USD", held=("AAPL", "VOO")))
    payload = json.loads(chain.calls[0][-1].content)
    for part in ("prices", "features", "asset_classes", "held"):
        assert set(payload[part]) == {"AAPL"}, part


def test_only_core_symbols_priced_skips_the_llm():
    chain = FakeChain(error=AssertionError("must not be invoked"))
    with capture_logs() as logs:
        assert _analyst(chain)(_state("VOO", "ETH-USD")) == []
    # The normal path outside the NYSE session: info, not a warning every cycle.
    (skipped,) = [e for e in logs if e["event"] == "analyst_skipped"]
    assert skipped["log_level"] == "info"


def test_a_flat_account_is_shown_as_holding_nothing_and_sells_are_limited_to_holdings():
    """Nothing held and AAPL below its trend: the analyst is told it holds nothing,
    and the rules allow a sell only of a holding."""
    chain = FakeChain(AnalystResponse(proposals=[]))
    state = _state("AAPL")
    state["features"]["AAPL"] = {"trend_up": 0.0}
    _analyst(chain)(state)

    (messages,) = chain.calls
    assert json.loads(messages[-1].content)["held"] == []
    system = messages[0].content
    assert 'When a symbol listed in held has trend_up 0, propose "sell"' in system
    assert 'Never propose "sell" for a symbol not listed in held or whose trend_up is 1' in system
    assert 'Only propose "buy" for a symbol not listed in held' in system
    assert 'trend_up is 1 and the symbol is not in held, lean toward "buy"' in system


def test_held_symbols_outside_the_priced_swing_set_are_not_shown():
    """Only holdings that are in the snapshot's prices are listed."""
    chain = FakeChain(AnalystResponse(proposals=[]))
    _analyst(chain)(_state("AAPL", held=("MSFT", "ETH-USD")))
    assert json.loads(chain.calls[0][-1].content)["held"] == []


def test_unknown_holdings_skip_the_llm():
    """The fetch step leaves held out when a position read failed: no guessing."""
    chain = FakeChain(error=AssertionError("must not be invoked"))
    state = _state("AAPL")
    del state["market_data"]["held"]
    with capture_logs() as logs:
        assert _analyst(chain)(state) == []
    assert chain.calls == []
    (skipped,) = [e for e in logs if e["event"] == "analyst_skipped"]
    assert skipped["reason"] == "holdings unknown"
