"""Per-symbol indicator snapshots from stored daily candles, for the analyst prompt.

An indicator whose lookback is not yet filled is absent from the snapshot rather
than None, so the analyst sees nothing instead of a null to misread.
"""

import math

import pandas as pd

from trading_agent.data.models import Candle
from trading_agent.features import indicators

MA_WINDOWS = (10, 20, 50, 200)
RSI_PERIOD = 14
ATR_PERIOD = 14
VWAP_WINDOW = 20
# 200-bar MAs need 200 bars; +60 slack keeps MACD/RSI smoothing warm.
CANDLE_LIMIT = 260


def snapshot(candles: list[Candle]) -> dict[str, float]:
    """The latest bar's indicator values for one symbol; {} without candles.

    Duplicate timestamps keep the last candle, as risk/stops.wilder_atr does, so both
    see the same series.
    """
    if not candles:
        return {}
    by_ts = {c.ts: c for c in sorted(candles, key=lambda c: c.ts)}
    frame = pd.DataFrame(
        {
            "high": [float(c.high) for c in by_ts.values()],
            "low": [float(c.low) for c in by_ts.values()],
            "close": [float(c.close) for c in by_ts.values()],
            "volume": [float(c.volume) for c in by_ts.values()],
        }
    )
    close = frame["close"]

    # float(): numpy.float64 must not leak into state — the checkpoint serializer refuses it.
    values: dict[str, float] = {
        "close": float(close.iloc[-1]),
        "volume": float(frame["volume"].iloc[-1]),
    }
    values["log_return_1d"] = _last(indicators.log_returns(close))
    for window in MA_WINDOWS:
        values[f"sma_{window}"] = _last(indicators.sma(close, window))
        values[f"ema_{window}"] = _last(indicators.ema(close, window))
    values[f"rsi_{RSI_PERIOD}"] = _last(indicators.rsi(close, RSI_PERIOD))

    macd_frame = indicators.macd(close)
    values["macd"] = _last(macd_frame["macd"])
    values["macd_signal"] = _last(macd_frame["signal"])
    values["macd_hist"] = _last(macd_frame["hist"])

    bands = indicators.bollinger(close)
    values["bb_upper"] = _last(bands["upper"])
    values["bb_mid"] = _last(bands["mid"])
    values["bb_lower"] = _last(bands["lower"])

    values[f"atr_{ATR_PERIOD}"] = _last(
        indicators.atr(frame["high"], frame["low"], close, ATR_PERIOD)
    )
    values["obv"] = _last(indicators.obv(close, frame["volume"]))
    values[f"vwap_{VWAP_WINDOW}"] = _last(
        indicators.vwap(frame["high"], frame["low"], close, frame["volume"], VWAP_WINDOW)
    )

    return {k: round(v, 6) for k, v in values.items() if v is not None}


def trend_up(complete: list[Candle]) -> float | None:
    """1.0 when the close is above its 50-bar average and the 20-bar is above the 50, else 0.0.

    None under 50 bars. Pass complete bars only: today's would flip the label intraday.
    """
    by_ts = {c.ts: c for c in sorted(complete, key=lambda c: c.ts)}
    if len(by_ts) < 50:
        return None
    close = pd.Series([float(c.close) for c in by_ts.values()])
    sma20, sma50 = _last(indicators.sma(close, 20)), _last(indicators.sma(close, 50))
    if sma20 is None or sma50 is None:
        return None
    return 1.0 if close.iloc[-1] > sma50 and sma20 > sma50 else 0.0


def _last(series: pd.Series) -> float | None:
    """The last value, or None when NaN or infinite (inf would be invalid JSON for the analyst)."""
    value = series.iloc[-1]
    return None if value is None or not math.isfinite(value) else float(value)
