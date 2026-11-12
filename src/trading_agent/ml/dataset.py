"""Directional-model features and forward-return labels.

One function computes features for both training and in-cycle inference, so they
cannot drift. Every feature is scale-invariant (returns, ratios, normalized
distances), so one model pools symbols at very different prices, and every feature
at bar t reads bars <= t only (pinned by test). No feature uses volume: the
sources measure it differently (IEX's partial tape, venue-only crypto volume).
"""

import numpy as np
import pandas as pd

from trading_agent.data.models import Candle
from trading_agent.features import indicators

HORIZON_BARS = 5
RETURN_WINDOWS = (1, 5, 10, 20)
FEATURES = (
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_10",
    "vol_20",
    "rsi_14",
    "dist_sma_10",
    "dist_sma_20",
    "bb_pctb_20",
    "atr_pct_14",
    "macd_hist_pct",
    "range_pct",
    "is_crypto",
)


def feature_frame(candles: list[Candle]) -> pd.DataFrame:
    """One row per bar for one symbol: every FEATURES column plus "close", indexed by time.

    Candles may come in any order; duplicates keep the last. Rows whose lookbacks
    are not yet filled hold NaN for callers to drop.
    """
    columns = [*FEATURES, "close"]
    if not candles:
        return pd.DataFrame(columns=columns, dtype=float)
    by_ts = {c.ts: c for c in sorted(candles, key=lambda c: c.ts)}
    index = pd.DatetimeIndex(list(by_ts))
    high = pd.Series([float(c.high) for c in by_ts.values()], index=index)
    low = pd.Series([float(c.low) for c in by_ts.values()], index=index)
    close = pd.Series([float(c.close) for c in by_ts.values()], index=index)

    log_ret = indicators.log_returns(close)
    out = pd.DataFrame(index=index)
    with np.errstate(divide="ignore", invalid="ignore"):
        for window in RETURN_WINDOWS:
            out[f"ret_{window}"] = np.log(close / close.shift(window))
        out["vol_10"] = log_ret.rolling(10).std()
        out["vol_20"] = log_ret.rolling(20).std()
        out["rsi_14"] = indicators.rsi(close, 14) / 100
        out["dist_sma_10"] = close / indicators.sma(close, 10) - 1
        out["dist_sma_20"] = close / indicators.sma(close, 20) - 1
        bands = indicators.bollinger(close, 20)
        width = bands["upper"] - bands["lower"]
        out["bb_pctb_20"] = (close - bands["lower"]) / width.where(width != 0)
        out["atr_pct_14"] = indicators.atr(high, low, close, 14) / close
        out["macd_hist_pct"] = indicators.macd(close)["hist"] / close
        out["range_pct"] = (high - low) / close
    out["is_crypto"] = float(candles[0].asset_class == "crypto")
    out["close"] = close
    # A zero close (bad tick) yields inf; treat it like an unfilled lookback.
    return out.replace([np.inf, -np.inf], np.nan)[columns]


def labelled_rows(candles: list[Candle], symbol: str) -> pd.DataFrame:
    """Training rows for one symbol: FEATURES, "label", "ts", "label_end" and "symbol".

    label is 1.0 when close[t + HORIZON_BARS] > close[t]. label_end is that future
    bar's time, so walk-forward splits can purge rows whose label reaches into the
    test window. Rows with an unfilled feature or no future bar are dropped.
    """
    frame = feature_frame(candles)
    future = frame["close"].shift(-HORIZON_BARS)
    frame["label"] = (future > frame["close"]).astype(float).where(future.notna())
    frame["ts"] = frame.index
    frame["label_end"] = frame.index.to_series().shift(-HORIZON_BARS)
    frame["symbol"] = symbol
    rows = frame.dropna(subset=[*FEATURES, "label"])
    return rows.drop(columns="close").reset_index(drop=True)
