"""Node 2, feature engine: indicator snapshots, news sentiment and Fear & Greed per symbol.

Features are advisory: a failure skips the symbol or empties the features, never
the cycle, and the risk gate reads its own candles. Sentiment keys appear only when
a headline was scored, so "no news" never reads as a neutral 0.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from trading_agent.features.engine import CANDLE_LIMIT, snapshot, trend_up
from trading_agent.graph.state import AgentState
from trading_agent.log import get_logger
from trading_agent.ml.sentiment import NEWS_JOB
from trading_agent.storage.repos import (
    JobRunRepository,
    NewsSentimentRepository,
    OhlcvRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from trading_agent.data.providers.fear_greed import FearGreedClient
    from trading_agent.graph.deps import CycleDeps

log = get_logger(__name__)

SENTIMENT_WINDOW = timedelta(hours=24)
# The worker's job runs every 10 minutes; no successful run for this long means it stopped.
SENTIMENT_STALE_AFTER = timedelta(hours=1)
# Published daily at 00:00 UTC; yesterday's reading is normal early in the day.
FEAR_GREED_MAX_AGE_DAYS = 1


def compute_features(state: AgentState, *, deps: CycleDeps) -> dict:
    """Return {"features": {symbol: {name: float}}} for every priced symbol that has candles.

    trend_up (1.0 or 0.0) is computed on complete bars only, so it holds all day like
    the backtest's daily decision, and is absent under 50 complete bars.
    """
    started = state.get("cycle_started_at") or datetime.now(UTC)
    today = started.replace(hour=0, minute=0, second=0, microsecond=0)
    market = state.get("market_data") or {}
    prices = market.get("prices") or {}
    features: dict[str, dict[str, float]] = {}
    try:
        with deps.session_factory() as session:
            repo = OhlcvRepository(session)
            for symbol in sorted(prices):
                try:
                    candles = repo.recent_daily_candles(symbol, limit=CANDLE_LIMIT)
                    values = snapshot(candles)
                    if values:
                        trend = trend_up([c for c in candles if c.ts < today])
                        if trend is not None:
                            values["trend_up"] = trend
                except Exception:
                    log.warning("features_symbol_failed", symbol=symbol, exc_info=True)
                    continue
                if values:
                    features[symbol] = values
                else:
                    log.warning("features_no_candles", symbol=symbol)
    except Exception:  # advisory data must never kill the cycle
        log.exception("features_unavailable", cycle_id=state.get("cycle_id"))
        return {"features": {}}
    try:
        # Own session: a failed candle read can leave the one above in an aborted transaction.
        with deps.session_factory() as session:
            _add_sentiment(session, features, state)
    except Exception:  # sentiment failing must never cost the price features
        log.warning("news_sentiment_unavailable", cycle_id=state.get("cycle_id"), exc_info=True)
    if deps.fear_greed is not None:
        _add_fear_greed(deps.fear_greed, features, state)

    log.info(
        "features_computed",
        cycle_id=state.get("cycle_id"),
        symbols={s: len(v) for s, v in features.items()},
    )
    return {"features": features}


def _add_sentiment(session: Session, features: dict, state: AgentState) -> None:
    """Add the last day's mean headline sentiment (-1..1) and count to each symbol's features.

    Warns when the worker's scoring job has not succeeded within SENTIMENT_STALE_AFTER.
    """
    now = state.get("cycle_started_at") or datetime.now(UTC)
    last_run = JobRunRepository(session).last_success(NEWS_JOB)
    if last_run is None or now - last_run > SENTIMENT_STALE_AFTER:
        log.warning("news_sentiment_stale", last_success_at=last_run)
    repo = NewsSentimentRepository(session)
    for symbol, values in features.items():
        mean, count = repo.summary(symbol, now - SENTIMENT_WINDOW)
        if count:
            values["news_sentiment_24h"] = round(mean, 4)
            values["news_count_24h"] = float(count)


def _add_fear_greed(client: FearGreedClient, features: dict, state: AgentState) -> None:
    """Add the market-wide Fear & Greed reading (0-100) to every crypto symbol's features.

    A failed fetch, or a reading too old or dated in the future, adds nothing and warns.
    """
    classes = (state.get("market_data") or {}).get("asset_classes") or {}
    crypto = [s for s in features if classes.get(s) == "crypto"]
    if not crypto:
        return
    cycle_id = state.get("cycle_id")
    try:
        value, day = client.latest()
    except Exception as e:
        log.warning("fear_greed_unavailable", cycle_id=cycle_id, reason=type(e).__name__)
        return
    today = (state.get("cycle_started_at") or datetime.now(UTC)).date()
    # A future-dated reading is a bad timestamp or clock skew, not fresh data.
    if not 0 <= (today - day).days <= FEAR_GREED_MAX_AGE_DAYS:
        log.warning("fear_greed_stale", cycle_id=cycle_id, day=day.isoformat())
        return
    for symbol in crypto:
        features[symbol]["crypto_fear_greed"] = float(value)
