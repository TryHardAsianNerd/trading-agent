"""Layer 1 tests: Wilder ATR, ATR stop, trailing stop, ratchet."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_agent.data.models import Candle
from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.stops import (
    atr_stop_price,
    effective_stop,
    trailing_stop_price,
    wilder_atr,
)

LIMITS = RiskLimits()
# The live limits have no trail (D-032); these tests pin the trail mechanism at 3%.
TRAIL = RiskLimits(trailing_stop_pct=Decimal("0.03"))
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _candle(i: int, high: str, low: str, close: str) -> Candle:
    return Candle(
        symbol="X",
        asset_class="stock",
        ts=T0 + timedelta(days=i),
        open=Decimal(close),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(1),
        source="test",
    )


def _flat_candles(n: int) -> list[Candle]:
    """Every day: high 101, low 99, close 100 → true range is exactly 2."""
    return [_candle(i, "101", "99", "100") for i in range(n)]


def test_atr_constant_true_range_is_exact():
    assert wilder_atr(_flat_candles(15)) == Decimal(2)


def test_atr_wilder_smoothing_converges_toward_new_regime():
    # 15 flat candles (TR=2), then a volatile one with TR=16 (high 108, low 92).
    candles = [*_flat_candles(15), _candle(15, "108", "92", "100")]
    atr = wilder_atr(candles)
    expected = (Decimal(2) * 13 + Decimal(16)) / 14  # one Wilder step from seed 2
    assert atr == expected
    assert Decimal(2) < atr < Decimal(16)


def test_atr_gap_uses_previous_close():
    # Day 1 closes 100; day 2 gaps to high 121 low 120: TR = |121 - 100| = 21.
    candles = [_candle(0, "101", "99", "100"), _candle(1, "121", "120", "120.5")]
    assert wilder_atr(candles, period=1) == Decimal(21)


def test_atr_order_independent():
    candles = _flat_candles(20)
    shuffled = [candles[7], *candles[8:], *candles[:7]]
    assert wilder_atr(shuffled) == wilder_atr(candles)


def test_atr_insufficient_data_raises():
    with pytest.raises(ValueError, match="needs at least 15"):
        wilder_atr(_flat_candles(14))


def test_atr_bad_period():
    with pytest.raises(ValueError):
        wilder_atr(_flat_candles(15), period=0)


def test_atr_stop_price_formula():
    # N = 3.0 (D-032) → stop = 100 - 3x3 = 91
    assert atr_stop_price(Decimal(100), Decimal(3), LIMITS) == Decimal(91)


@pytest.mark.parametrize("entry,atr", [(Decimal(0), Decimal(1)), (Decimal(100), Decimal(0))])
def test_atr_stop_rejects_nonpositive_inputs(entry, atr):
    with pytest.raises(ValueError):
        atr_stop_price(entry, atr, LIMITS)


def test_trailing_stop_is_3_pct_below_high_water():
    assert trailing_stop_price(Decimal(200), TRAIL) == Decimal("194.00")


def test_live_limits_have_no_fixed_trail():
    """The live limits have no fixed trail (D-032)."""
    assert LIMITS.trailing_stop_pct is None
    with pytest.raises(ValueError):
        trailing_stop_price(Decimal(200), LIMITS)
    # With no trail the effective stop is the ATR stop, however far price has run.
    assert effective_stop(Decimal(100), Decimal(2), Decimal(150), LIMITS) == Decimal(94)


def test_effective_stop_ratchets_up_with_price():
    entry, atr = Decimal(100), Decimal(2)
    # At entry: ATR stop 94 vs trail 97 → trail wins already.
    assert effective_stop(entry, atr, entry, TRAIL) == Decimal("97.00")
    # Price ran to 120: trail 116.4 dominates.
    assert effective_stop(entry, atr, Decimal(120), TRAIL) == Decimal("116.40")


def test_effective_stop_never_below_initial():
    entry, atr = Decimal(100), Decimal(10)  # ATR stop 70, trail at entry 97
    assert effective_stop(entry, atr, entry, TRAIL) == Decimal("97.00")
    # Even if the "high" were somehow below entry, stop never drops under ATR stop.
    assert effective_stop(entry, atr, Decimal(70), TRAIL) == Decimal(70)
