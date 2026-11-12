"""Technical indicators as pure pandas functions on ascending Series.

Floats are fine here (indicators are advisory, never money). Outputs are NaN until
the lookback fills. RSI and ATR use the same Wilder smoothing as risk/stops.wilder_atr.
"""

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    """ln(close_t / close_{t-1}); a bad zero close gives -inf, which the engine drops."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(close / close.shift(1))


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average over window bars."""
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (alpha = 2/(span+1)), NaN until span bars exist."""
    values = close.ewm(span=span, adjust=False).mean()
    return values.mask(_unfilled(close, span))


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI in [0, 100], textbook SMA-seeded Wilder convention.

    A flat window is NaN rather than 100, so a halted symbol never reads as overbought.
    """
    delta = close.diff()
    avg_gain = _wilder_smooth(delta.clip(lower=0), period)
    avg_loss = _wilder_smooth(-delta.clip(upper=0), period)
    rs = avg_gain / avg_loss.where(avg_loss != 0)
    values = (100 - 100 / (1 + rs)).where(avg_loss != 0, other=100.0)
    values = values.mask((avg_loss == 0) & (avg_gain == 0))
    return values.mask(_unfilled(close, period + 1))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram as columns "macd", "signal", "hist"."""
    line = ema(close, fast) - ema(close, slow)
    # Mask the signal until its own window fills too: an EMA of one macd value is noise.
    sig = line.ewm(span=signal, adjust=False).mean().mask(_unfilled(close, slow + signal - 1))
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands ("upper", "mid", "lower") with population std, as charts compute them."""
    mid = sma(close, window)
    std = close.rolling(window).std(ddof=0)
    return pd.DataFrame({"upper": mid + num_std * std, "mid": mid, "lower": mid - num_std * std})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing; NaN until period+1 bars exist."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    true_range.iloc[0] = np.nan  # no previous close: first bar has no TR
    return _wilder_smooth(true_range, period)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by each close-to-close move, from 0."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling volume-weighted typical price over window bars; NaN while volume is zero."""
    typical = (high + low + close) / 3
    notional = (typical * volume).rolling(window).sum()
    total_volume = volume.rolling(window).sum()
    return notional / total_volume.where(total_volume != 0)


def _wilder_smooth(values: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing: an SMA seed over values[1..period], then (prev*(n-1) + value)/n."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if len(values) <= period:
        return out
    running = values.iloc[1 : period + 1].mean()
    out.iloc[period] = running
    for i in range(period + 1, len(values)):
        running = (running * (period - 1) + values.iloc[i]) / period
        out.iloc[i] = running
    return out


def _unfilled(series: pd.Series, window: int) -> pd.Series:
    """True while fewer than window bars have been seen."""
    return pd.Series(range(len(series)), index=series.index) < window - 1
