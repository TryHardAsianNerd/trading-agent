"""Liquidity filter unit tests (D-022) — RISK_RULES "high-liquidity instruments only".

Exhaustive per the risk-module rule: window arithmetic, the per-asset-class
volume-unit convention, both thresholds, boundaries to the cent, and
Decimal-only arithmetic.
"""

import dataclasses
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_agent.data.models import Candle
from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.liquidity import average_dollar_volume, is_liquid

LIMITS = RiskLimits()
WINDOW = LIMITS.liquidity_window_days
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _candles(
    n: int,
    close: Decimal = Decimal(100),
    volume: Decimal = Decimal(1),
    asset_class: str = "stock",
    start: int = 0,
) -> list[Candle]:
    """n daily candles with a fixed close and volume, one per day from T0+start."""
    return [
        Candle(
            symbol="SYM",
            asset_class=asset_class,
            ts=T0 + timedelta(days=start + i),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=volume,
            source="test",
        )
        for i in range(n)
    ]


# --- average_dollar_volume: arithmetic -------------------------------------


def test_stock_dollar_volume_is_close_times_share_volume():
    avg = average_dollar_volume(
        _candles(WINDOW, close=Decimal(50), volume=Decimal(1000)), "stock", LIMITS
    )
    assert avg == Decimal(50_000)


def test_crypto_volume_is_already_usd_and_is_not_multiplied_by_close():
    """CoinGecko total_volumes is denominated in the vs_currency (USD)."""
    candles = _candles(
        WINDOW, close=Decimal(60_000), volume=Decimal(1_000_000), asset_class="crypto"
    )
    assert average_dollar_volume(candles, "crypto", LIMITS) == Decimal(1_000_000)
    # The same rows read as a stock would be 60,000x larger — the unit bug this guards.
    assert average_dollar_volume(candles, "stock", LIMITS) == Decimal(60_000_000_000)


def test_average_uses_only_the_most_recent_window():
    """Older bars outside the window must not drag the average up."""
    old = _candles(10, close=Decimal(100), volume=Decimal(10_000_000), start=0)
    recent = _candles(WINDOW, close=Decimal(100), volume=Decimal(1_000), start=10)
    assert average_dollar_volume(old + recent, "stock", LIMITS) == Decimal(100_000)


def test_average_is_a_true_mean_of_varying_bars():
    candles = _candles(WINDOW, close=Decimal(10), volume=Decimal(100))
    # Replace the newest bar with a 20x day; the mean moves by 19/WINDOW of a bar.
    hot = candles[-1].model_copy(update={"volume": Decimal(2_000)})
    candles = [*candles[:-1], hot]
    expected = (Decimal(1_000) * (WINDOW - 1) + Decimal(20_000)) / WINDOW
    assert average_dollar_volume(candles, "stock", LIMITS) == expected


def test_unsorted_candles_are_ordered_before_the_window_is_taken():
    ordered = _candles(WINDOW + 5, close=Decimal(100), volume=Decimal(1_000))
    hot = ordered[0].model_copy(update={"volume": Decimal(10_000_000)})  # oldest bar, excluded
    shuffled = [hot, *reversed(ordered[1:])]
    assert average_dollar_volume(shuffled, "stock", LIMITS) == Decimal(100_000)


def test_duplicate_timestamps_are_deduped_last_wins():
    """Duplicated ingestion rows must not double-count a session (cf. wilder_atr)."""
    clean = _candles(WINDOW, close=Decimal(100), volume=Decimal(1_000))
    assert average_dollar_volume(clean + clean, "stock", LIMITS) == average_dollar_volume(
        clean, "stock", LIMITS
    )


def test_result_is_decimal_not_float():
    avg = average_dollar_volume(_candles(WINDOW, volume=Decimal(1_000)), "stock", LIMITS)
    assert type(avg) is Decimal


def test_decimal_arithmetic_is_exact_where_float_would_drift():
    candles = _candles(WINDOW, close=Decimal("0.1"), volume=Decimal(3))
    assert average_dollar_volume(candles, "stock", LIMITS) == Decimal("0.3")


# --- average_dollar_volume: insufficient history ---------------------------


def test_fewer_than_window_candles_raises():
    with pytest.raises(ValueError, match="at least 20 distinct-timestamp candles"):
        average_dollar_volume(_candles(WINDOW - 1, volume=Decimal(1_000)), "stock", LIMITS)


def test_exactly_window_candles_is_enough():
    assert average_dollar_volume(_candles(WINDOW, volume=Decimal(1_000)), "stock", LIMITS) > 0


def test_duplicates_do_not_fake_up_the_bar_count():
    """10 distinct sessions duplicated twice are still only 10 sessions."""
    ten = _candles(10, volume=Decimal(1_000))
    with pytest.raises(ValueError, match="got 10"):
        average_dollar_volume(ten + ten, "stock", LIMITS)


def test_empty_candles_raises():
    with pytest.raises(ValueError):
        average_dollar_volume([], "stock", LIMITS)


# --- is_liquid: thresholds and boundaries ----------------------------------


def _stock_at(avg: Decimal) -> list[Candle]:
    """Stock candles whose average daily dollar volume is exactly `avg`."""
    return _candles(WINDOW, close=Decimal(1), volume=avg)


def test_liquid_stock_passes():
    liquid, detail = is_liquid(_stock_at(Decimal(500_000_000)), "stock", LIMITS)
    assert liquid
    assert detail == "20d avg dollar volume $500,000,000.00 vs stock minimum $10,000,000.00"


def test_stock_exactly_at_the_threshold_is_liquid():
    at = LIMITS.min_avg_dollar_volume_stock
    liquid, _ = is_liquid(_stock_at(at), "stock", LIMITS)
    assert liquid


def test_stock_one_cent_below_the_threshold_is_illiquid():
    """A near miss must be distinguishable from an approval in the audit line."""
    below = LIMITS.min_avg_dollar_volume_stock - Decimal("0.01")
    liquid, detail = is_liquid(_stock_at(below), "stock", LIMITS)
    assert not liquid
    assert detail.startswith("20d avg dollar volume $9,999,999.99 ")


def test_illiquid_detail_carries_the_measured_figure():
    liquid, detail = is_liquid(_stock_at(Decimal(1234)), "stock", LIMITS)
    assert not liquid
    assert detail.startswith("20d avg dollar volume $1,234.00 ")


def test_zero_volume_is_illiquid_not_an_error():
    liquid, _ = is_liquid(_stock_at(Decimal(0)), "stock", LIMITS)
    assert not liquid


def test_crypto_threshold_is_separate_from_the_stock_one():
    """A level that clears stocks but not crypto proves the thresholds are per class."""
    between = (LIMITS.min_avg_dollar_volume_stock + LIMITS.min_avg_dollar_volume_crypto) / 2
    assert LIMITS.min_avg_dollar_volume_stock < between < LIMITS.min_avg_dollar_volume_crypto
    stock_rows = _candles(WINDOW, close=Decimal(1), volume=between)
    crypto_rows = _candles(WINDOW, close=Decimal(1), volume=between, asset_class="crypto")
    assert is_liquid(stock_rows, "stock", LIMITS)[0]
    assert not is_liquid(crypto_rows, "crypto", LIMITS)[0]


def test_crypto_exactly_at_the_threshold_is_liquid():
    at = LIMITS.min_avg_dollar_volume_crypto
    rows = _candles(WINDOW, close=Decimal(60_000), volume=at, asset_class="crypto")
    assert is_liquid(rows, "crypto", LIMITS)[0]


def test_crypto_one_cent_below_the_threshold_is_illiquid():
    below = LIMITS.min_avg_dollar_volume_crypto - Decimal("0.01")
    rows = _candles(WINDOW, close=Decimal(60_000), volume=below, asset_class="crypto")
    assert not is_liquid(rows, "crypto", LIMITS)[0]


def test_is_liquid_propagates_insufficient_history():
    with pytest.raises(ValueError, match="distinct-timestamp candles"):
        is_liquid(_stock_at(Decimal(500_000_000))[:5], "stock", LIMITS)


def test_thresholds_are_decimal_constants_on_frozen_limits():
    """Risk parameters, not settings: frozen, Decimal, never env- or LLM-supplied."""
    assert type(LIMITS.min_avg_dollar_volume_stock) is Decimal
    assert type(LIMITS.min_avg_dollar_volume_crypto) is Decimal
    with pytest.raises(dataclasses.FrozenInstanceError):
        LIMITS.min_avg_dollar_volume_stock = Decimal(1)
