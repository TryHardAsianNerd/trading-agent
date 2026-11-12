"""Known-value tests for the hand-rolled Tier-1 indicators (D-015).

Each indicator is checked against small hand-computed cases (worked in the
comments), boundary behavior (flat/all-up/all-down), and lookback masking.
ATR is additionally pinned to parity with risk/stops.wilder_atr so the two
implementations cannot drift.
"""

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from trading_agent.data.models import Candle
from trading_agent.features import indicators
from trading_agent.features.engine import CANDLE_LIMIT, snapshot, trend_up
from trading_agent.risk.stops import wilder_atr


def _series(*values) -> pd.Series:
    return pd.Series([float(v) for v in values])


def _candle(i: int, o="100", h="101", lo="99", c="100", v="1000", symbol="TEST") -> Candle:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return Candle(
        symbol=symbol,
        asset_class="stock",
        ts=base + timedelta(days=i),
        open=Decimal(o),
        high=Decimal(h),
        low=Decimal(lo),
        close=Decimal(c),
        volume=Decimal(v),
        source="test",
    )


# --- log returns ---


def test_log_returns_known_value():
    result = indicators.log_returns(_series(1, 2, 4))
    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(math.log(2))
    assert result.iloc[2] == pytest.approx(math.log(2))


# --- SMA / EMA ---


def test_sma_known_values_and_masking():
    result = indicators.sma(_series(1, 2, 3, 4, 5), window=3)
    assert math.isnan(result.iloc[1])  # window not filled
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_known_values():
    # span=2 -> alpha=2/3: ema = [1, 1+2/3(2-1)=5/3, 5/3+2/3(3-5/3)=23/9]
    result = indicators.ema(_series(1, 2, 3), span=2)
    assert result.iloc[1] == pytest.approx(5 / 3)
    assert result.iloc[2] == pytest.approx(23 / 9)


def test_ema_masks_unfilled_window():
    # A 2-bar "EMA(200)" is noise, not signal: masked to NaN.
    result = indicators.ema(_series(1, 2), span=200)
    assert math.isnan(result.iloc[-1])


# --- RSI ---


def test_rsi_boundaries():
    up = indicators.rsi(_series(*range(1, 20)), period=14)
    assert up.iloc[-1] == pytest.approx(100.0)
    down = indicators.rsi(_series(*range(20, 1, -1)), period=14)
    assert down.iloc[-1] == pytest.approx(0.0)


def test_rsi_known_value_hand_computed():
    """period=2, closes [10, 11, 10.5, 11.5] -> deltas [+1, -0.5, +1].
    SMA seed over first 2 deltas: avg_gain=.5, avg_loss=.25. Wilder step on
    delta +1: avg_gain=(.5+1)/2=.75, avg_loss=(.25+0)/2=.125.
    RS = .75/.125 = 6 -> RSI = 100 - 100/7 = 85.714285..."""
    result = indicators.rsi(_series(10, 11, 10.5, 11.5), period=2)
    assert result.iloc[-1] == pytest.approx(100 - 100 / 7)


def test_rsi_matches_textbook_wilder_reference():
    """Pin the SEED convention: SMA over the first `period` deltas, then
    Wilder recursion (talib/TradingView). An ewm-from-bar-1 'Wilder' RSI
    diverges from this by up to ~29 points on 30-bar (crypto-length)
    histories — the reference below is the published recursion, written
    independently of the implementation."""
    closes = [100.0]
    for i in range(29):  # deterministic pseudo-random walk, 30 bars
        closes.append(closes[-1] + ((i * 7919) % 13 - 6) * 0.5)
    period = 14
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, lo in zip(gains[period:], losses[period:], strict=True):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + lo) / period
    expected = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    result = indicators.rsi(_series(*closes), period=period)
    assert result.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_rsi_flat_series_is_masked_not_overbought():
    """No gains AND no losses: RS is 0/0, not infinity — a stablecoin must
    not read as 'RSI 100 overbought' and invite a spurious sell."""
    result = indicators.rsi(_series(*([100] * 20)), period=14)
    assert math.isnan(result.iloc[-1])


def test_rsi_masked_until_period_plus_one_bars():
    result = indicators.rsi(_series(10, 11, 12), period=14)
    assert math.isnan(result.iloc[-1])


# --- MACD ---


def test_macd_flat_series_is_zero():
    frame = indicators.macd(_series(*([50] * 40)))
    assert frame["macd"].iloc[-1] == pytest.approx(0.0)
    assert frame["signal"].iloc[-1] == pytest.approx(0.0)
    assert frame["hist"].iloc[-1] == pytest.approx(0.0)


def test_macd_uptrend_is_positive():
    frame = indicators.macd(_series(*range(1, 60)))
    assert frame["macd"].iloc[-1] > 0
    assert frame["macd"].iloc[-1] == pytest.approx(
        frame["signal"].iloc[-1] + frame["hist"].iloc[-1]
    )


def test_macd_masked_until_slow_window_fills():
    frame = indicators.macd(_series(*range(1, 20)))  # 19 bars < slow 26
    assert math.isnan(frame["macd"].iloc[-1])


def test_macd_signal_masked_until_its_own_window_fills():
    """An unmasked signal would ship as an 'EMA(9)' of one macd value at bar
    26 (hist = exactly 0: 'crossover imminent') — the same noise the ema()
    mask exists to suppress. Signal/hist hold back until slow+signal-1."""
    bars_30 = indicators.macd(_series(*range(1, 31)))  # macd yes, signal not yet
    assert not math.isnan(bars_30["macd"].iloc[-1])
    assert math.isnan(bars_30["signal"].iloc[-1])
    assert math.isnan(bars_30["hist"].iloc[-1])
    bars_34 = indicators.macd(_series(*range(1, 35)))  # 34 = 26 + 9 - 1: filled
    assert not math.isnan(bars_34["signal"].iloc[-1])


# --- Bollinger ---


def test_bollinger_flat_series_collapses_to_mid():
    bands = indicators.bollinger(_series(*([10] * 25)))
    assert bands["upper"].iloc[-1] == pytest.approx(10.0)
    assert bands["mid"].iloc[-1] == pytest.approx(10.0)
    assert bands["lower"].iloc[-1] == pytest.approx(10.0)


def test_bollinger_known_value_hand_computed():
    """window=2 on [1, 3]: mid=2, POPULATION std=sqrt(((1-2)^2+(3-2)^2)/2)=1
    (Bollinger's published definition, ddof=0): upper = 4, lower = 0."""
    bands = indicators.bollinger(_series(1, 3), window=2)
    assert bands["mid"].iloc[-1] == pytest.approx(2.0)
    assert bands["upper"].iloc[-1] == pytest.approx(4.0)
    assert bands["lower"].iloc[-1] == pytest.approx(0.0)


# --- ATR (parity with the risk engine) ---


def test_atr_matches_risk_engine_wilder_atr():
    """The feature ATR and risk/stops.wilder_atr implement the same Wilder
    definition — parity pinned so they can never drift apart."""
    candles = [
        _candle(i, h=str(101 + (i % 5)), lo=str(99 - (i % 3)), c=str(100 + (i % 4)))
        for i in range(30)
    ]
    expected = float(wilder_atr(candles))
    result = indicators.atr(
        pd.Series([float(c.high) for c in candles]),
        pd.Series([float(c.low) for c in candles]),
        pd.Series([float(c.close) for c in candles]),
    )
    assert result.iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_atr_masked_until_period_plus_one_bars():
    result = indicators.atr(_series(2, 2), _series(1, 1), _series(1.5, 1.5), period=14)
    assert math.isnan(result.iloc[-1])


# --- OBV ---


def test_obv_hand_computed():
    """closes [1,2,2,1], volumes [10,20,30,40]: first bar 0 (no prior close),
    up +20, flat +0, down -40 -> cumulative [0, 20, 20, -20]."""
    result = indicators.obv(_series(1, 2, 2, 1), _series(10, 20, 30, 40))
    assert list(result) == [0.0, 20.0, 20.0, -20.0]


# --- VWAP ---


def test_vwap_hand_computed():
    """window=2; typical prices [(2+0+1)/3=1, (4+2+3)/3=3], volumes [10, 30]:
    VWAP = (1*10 + 3*30) / 40 = 2.5."""
    result = indicators.vwap(_series(2, 4), _series(0, 2), _series(1, 3), _series(10, 30), window=2)
    assert result.iloc[-1] == pytest.approx(2.5)


def test_vwap_zero_volume_window_is_nan():
    result = indicators.vwap(_series(2, 4), _series(0, 2), _series(1, 3), _series(0, 0), window=2)
    assert math.isnan(result.iloc[-1])


# --- engine snapshot ---


def test_snapshot_full_history_has_every_indicator():
    candles = [_candle(i, c=str(100 + (i % 7))) for i in range(CANDLE_LIMIT)]
    values = snapshot(candles)
    expected_keys = {
        "close",
        "volume",
        "log_return_1d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        "atr_14",
        "obv",
        "vwap_20",
        *{f"{kind}_{w}" for kind in ("sma", "ema") for w in (10, 20, 50, 200)},
    }
    assert set(values) == expected_keys
    # type() not isinstance(): numpy.float64 subclasses float but the serializer refuses it.
    assert all(type(v) is float for v in values.values())


def test_snapshot_short_crypto_history_omits_long_lookbacks():
    """The 30-day crypto cap: the long MAs must be ABSENT, not None or zero — the
    LLM should see nothing rather than a null to misread."""
    values = snapshot([_candle(i, c=str(100 + (i % 7))) for i in range(25)])
    for missing in ("sma_50", "sma_200", "ema_50", "ema_200", "macd"):
        assert missing not in values
    for present in ("close", "rsi_14", "atr_14", "sma_20", "obv", "vwap_20"):
        assert present in values


def test_snapshot_dedupes_candles_by_timestamp():
    base = [_candle(i, c="100") for i in range(30)]
    corrected = _candle(29, c="200")  # provider correction for the same bar
    values = snapshot([*base, corrected])
    assert values["close"] == 200.0
    assert values["sma_10"] == pytest.approx((9 * 100 + 200) / 10)


def test_snapshot_empty_candles_is_empty():
    assert snapshot([]) == {}


def test_snapshot_zero_close_never_emits_infinity():
    """A zero close (bad provider tick) makes log_returns emit -inf/+inf;
    json.dumps would render that as `Infinity` — invalid JSON for the LLM
    request. Non-finite values must be ABSENT, like unfilled lookbacks."""
    candles = [_candle(i, c="100") for i in range(19)] + [_candle(19, c="0")]
    values = snapshot(candles)
    assert "log_return_1d" not in values
    assert values["close"] == 0.0
    assert all(math.isfinite(v) for v in values.values())


# --- trend_up (D-038) ---


def _closes(values: list[float]) -> list[Candle]:
    return [_candle(i, c=str(v)) for i, v in enumerate(values)]


def test_trend_up_needs_fifty_complete_bars():
    assert trend_up(_closes([100.0] * 49)) is None
    assert trend_up(_closes([100.0] * 50)) == 0.0  # flat: the close equals the 50-day


def test_equal_averages_are_not_an_uptrend():
    """Close above both, but the 20-day exactly equals the 50-day: strictly above is required."""
    closes = [100.0] * 30 + [99.0] * 19 + [119.0]  # 20-day = 50-day = 100, close 119
    assert trend_up(_closes(closes)) == 0.0


def test_trend_up_is_the_price_over_the_50_day_and_the_20_day_over_the_50_day():
    rising = [float(v) for v in range(1, 61)]
    assert trend_up(_closes(rising)) == 1.0
    # Close still above the 50-day but under the 20-day: an uptrend all the same.
    assert trend_up(_closes([*rising[:-1], 45.0])) == 1.0
    # Close under the 50-day: no.
    assert trend_up(_closes([*rising[:-1], 30.0])) == 0.0
    # 20-day under the 50-day (a fall), close above both: no.
    falling = [float(v) for v in range(60, 0, -1)]
    assert trend_up(_closes([*falling[:-1], 100.0])) == 0.0
