"""Node 3, ML predictor: the directional model's probability that each symbol closes higher.

Predictions are logged only (D-032): neither the analyst nor the risk gate reads
them, so every failure degrades to no predictions and never stops the cycle. The
newest model version loads each cycle and scores complete bars only, as it was trained.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from trading_agent.data.models import Candle
from trading_agent.features.engine import CANDLE_LIMIT
from trading_agent.graph.state import AgentState
from trading_agent.log import get_logger
from trading_agent.ml.dataset import FEATURES, feature_frame
from trading_agent.ml.model import MIN_OOS_AUC, is_servable, load_latest
from trading_agent.storage.repos import OhlcvRepository

if TYPE_CHECKING:
    from trading_agent.graph.deps import CycleDeps

log = get_logger(__name__)

# Training rows sit on years of warm EMA/Wilder state; MACD and RSI converge by ~60 bars.
MIN_HISTORY_BARS = 60
# Calendar days: crypto closes daily; stocks span a weekend plus a holiday.
MAX_BAR_AGE_DAYS = {"crypto": 1, "stock": 4}
# The worker retrains weekly; older than this means the retrain stopped running.
MAX_MODEL_AGE = timedelta(days=8)


def predict(state: AgentState, *, deps: CycleDeps) -> dict:
    """Score every priced symbol with the newest servable model.

    Returns {"predictions": {}} without one, else {"predictions": {"model": {version,
    horizon_bars, oos_auc, base_rate}, "by_symbol": {symbol: {prob_up, as_of}}}}.
    """
    cycle_id = state.get("cycle_id")
    started = state.get("cycle_started_at") or datetime.now(UTC)
    try:
        model = load_latest(deps.model_dir)
        servable = model is None or is_servable(model.meta["walk_forward"])
        age = started - datetime.fromisoformat(model.meta["trained_at"]) if model else None
    except Exception:  # corrupt/incompatible artifact: retrain, never crash the cycle
        log.warning("ml_model_unloadable", cycle_id=cycle_id, exc_info=True)
        return {"predictions": {}}
    if model is None:
        log.warning(
            "ml_model_missing", cycle_id=cycle_id, note="run: python -m trading_agent.ml.train"
        )
        return {"predictions": {}}
    if not servable:
        log.warning(
            "ml_model_below_skill_floor",
            cycle_id=cycle_id,
            version=model.version,
            walk_forward=model.meta["walk_forward"],
            min_oos_auc=MIN_OOS_AUC,
        )
        return {"predictions": {}}

    if age > MAX_MODEL_AGE:
        # Still served: an old model is no worse than it was, but the weekly retrain has stopped.
        log.warning("ml_model_stale", cycle_id=cycle_id, version=model.version, age_days=age.days)
    prices = (state.get("market_data") or {}).get("prices") or {}
    today = datetime.combine(started.astimezone(UTC).date(), time(0), UTC)
    latest: dict[str, pd.Series] = {}
    try:
        with deps.session_factory() as session:
            repo = OhlcvRepository(session)
            for symbol in sorted(prices):
                candles = repo.recent_daily_candles(symbol, limit=CANDLE_LIMIT)
                row = _latest_row([c for c in candles if c.ts < today], symbol, today)
                if row is not None:
                    latest[symbol] = row
        probs = model.prob_up(pd.DataFrame(list(latest.values()))) if latest else []
    except Exception:  # advisory data must never kill the cycle
        log.exception("ml_predictions_unavailable", cycle_id=cycle_id)
        return {"predictions": {}}

    by_symbol = {
        symbol: {"prob_up": round(p, 4), "as_of": row.name.date().isoformat()}
        for (symbol, row), p in zip(latest.items(), probs, strict=True)
    }
    predictions = {
        "model": {
            "version": model.version,
            "horizon_bars": model.meta["horizon_bars"],
            "oos_auc": model.oos_auc,
            "base_rate": float(model.meta["walk_forward"]["base_rate"]),
        },
        "by_symbol": by_symbol,
    }
    log.info("ml_predictions", cycle_id=cycle_id, **predictions)
    return {"predictions": predictions}


def _latest_row(complete: list[Candle], symbol: str, today: datetime) -> pd.Series | None:
    """The newest complete bar's feature row, or None (warned) if it is unfit to score.

    Unfit means too little history, a stale newest bar, or missing or failed features.
    """
    if len(complete) < MIN_HISTORY_BARS:
        log.warning("ml_prediction_skipped", symbol=symbol, reason="history_too_short")
        return None
    newest = max(complete, key=lambda c: c.ts)
    if (today - newest.ts).days > MAX_BAR_AGE_DAYS[newest.asset_class]:
        log.warning(
            "ml_prediction_skipped", symbol=symbol, reason="stale_bars", as_of=newest.ts.date()
        )
        return None
    try:
        row = feature_frame(complete).iloc[-1]
    except Exception:
        log.warning("ml_prediction_skipped", symbol=symbol, reason="features_failed", exc_info=True)
        return None
    if row[list(FEATURES)].isna().any():
        log.warning("ml_prediction_skipped", symbol=symbol, reason="features_incomplete")
        return None
    return row
