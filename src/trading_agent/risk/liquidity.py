"""Liquidity filter: "high-liquidity instruments only" (RISK_RULES.md).

Average daily dollar volume over a frozen window against a per-asset-class minimum.
Volume units differ: stock candles carry a share count (dollar volume is close x
volume), crypto candles already store USD. Multiplying crypto by the close would turn
the filter into a no-op, so a crypto provider reporting coin quantity must revisit this.
The crypto floor is calibrated on market-wide volume; crypto is core only, so no crypto
order reaches the gate.

Stock bars are the Alpaca IEX feed, 2.5-5% of the consolidated tape, so the stock
threshold is calibrated on IEX volume and is not comparable to a published ADV.
A partial bar from an open session lowers the average, erring toward rejecting.
"""

from decimal import Decimal

from trading_agent.data.models import AssetClass, Candle
from trading_agent.risk.limits import RiskLimits


def average_dollar_volume(
    candles: list[Candle], asset_class: AssetClass, limits: RiskLimits
) -> Decimal:
    """Average daily dollar volume over the last `liquidity_window_days`; ValueError if short.

    Duplicate timestamps keep the last candle, so a duplicated row cannot double-count.
    """
    window = limits.liquidity_window_days
    by_ts = {c.ts: c for c in sorted(candles, key=lambda c: c.ts)}
    if len(by_ts) < window:
        raise ValueError(
            f"liquidity needs at least {window} distinct-timestamp candles, got {len(by_ts)}"
        )
    recent = list(by_ts.values())[-window:]
    if asset_class == "crypto":
        total = sum((c.volume for c in recent), Decimal(0))
    else:
        total = sum((c.close * c.volume for c in recent), Decimal(0))
    return total / window


def is_liquid(
    candles: list[Candle], asset_class: AssetClass, limits: RiskLimits
) -> tuple[bool, str]:
    """(liquid enough to open a position, detail with the measured figure and the minimum).

    ValueError when there is too little history to measure.
    """
    average = average_dollar_volume(candles, asset_class, limits)
    minimum = (
        limits.min_avg_dollar_volume_crypto
        if asset_class == "crypto"
        else limits.min_avg_dollar_volume_stock
    )
    # Cents, so a near-miss rejection is distinguishable from an approval in the audit row.
    detail = (
        f"{limits.liquidity_window_days}d avg dollar volume ${average:,.2f} "
        f"vs {asset_class} minimum ${minimum:,.2f}"
    )
    return average >= minimum, detail
