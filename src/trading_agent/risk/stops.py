"""Layer 1 stops: Wilder ATR, the ATR-based initial stop, and the optional trail."""

import itertools
from decimal import Decimal

from trading_agent.data.models import Candle
from trading_agent.risk.limits import RiskLimits

ATR_PERIOD = 14


def wilder_atr(candles: list[Candle], period: int = ATR_PERIOD) -> Decimal:
    """Wilder-smoothed ATR over daily candles; ValueError under period + 1 distinct bars."""
    if period < 1:
        raise ValueError("ATR period must be >= 1")
    if len(candles) < period + 1:
        raise ValueError(f"ATR({period}) needs at least {period + 1} candles, got {len(candles)}")
    # Duplicate rows would inject zero true ranges and deflate the ATR, so last wins.
    by_ts = {c.ts: c for c in sorted(candles, key=lambda c: c.ts)}
    ordered = list(by_ts.values())
    if len(ordered) < period + 1:
        raise ValueError(
            f"ATR({period}) needs at least {period + 1} distinct-timestamp candles, "
            f"got {len(ordered)}"
        )
    true_ranges: list[Decimal] = []
    for prev, cur in itertools.pairwise(ordered):
        true_ranges.append(
            max(
                cur.high - cur.low,
                abs(cur.high - prev.close),
                abs(cur.low - prev.close),
            )
        )
    atr = sum(true_ranges[:period], Decimal(0)) / period  # SMA seed
    for tr in true_ranges[period:]:  # Wilder smoothing
        atr = (atr * (period - 1) + tr) / period
    return atr


def atr_stop_price(entry_price: Decimal, atr: Decimal, limits: RiskLimits) -> Decimal:
    """Initial long stop, entry - N x ATR; may be <= 0 for extreme ATR, which callers reject."""
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if atr <= 0:
        raise ValueError("atr must be positive")
    return entry_price - limits.atr_multiplier * atr


def trailing_stop_price(highest_price_since_entry: Decimal, limits: RiskLimits) -> Decimal:
    """Trailing stop a fixed percentage below the high since entry; ValueError without a trail."""
    if limits.trailing_stop_pct is None:
        raise ValueError("these limits have no trailing stop")
    if highest_price_since_entry <= 0:
        raise ValueError("highest_price_since_entry must be positive")
    return highest_price_since_entry * (1 - limits.trailing_stop_pct)


def effective_stop(
    entry_price: Decimal,
    atr: Decimal,
    highest_price_since_entry: Decimal,
    limits: RiskLimits,
) -> Decimal:
    """The higher of the initial ATR stop and the trail, or the ATR stop when there is none.

    `atr` is the ATR at entry, which fixes the initial stop.
    """
    initial = atr_stop_price(entry_price, atr, limits)
    if limits.trailing_stop_pct is None:
        return initial
    return max(initial, trailing_stop_price(highest_price_since_entry, limits))
