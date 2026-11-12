"""Layer 2 fixed-fractional sizing and the Layer 4 cash reserve.

size = (equity x risk_pct) / (entry - stop), capped by the max position value and by
deployable cash; the 20% reserve is a hard floor no trade may dip below. Quantities are
never rounded here: the executor rounds lots, and only down.
"""

from decimal import Decimal

from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.models import PortfolioState


def risk_based_quantity(
    equity: Decimal, entry_price: Decimal, stop_price: Decimal, limits: RiskLimits
) -> Decimal:
    """Quantity such that hitting the stop loses exactly risk_per_trade_pct of equity.

    ValueError unless 0 < stop_price < entry_price.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if stop_price >= entry_price:
        raise ValueError("stop must be below entry for a long position")
    if stop_price <= 0:
        raise ValueError("stop_price must be positive (stop distance too wide for this entry)")
    risk_amount = equity * limits.risk_per_trade_pct
    return risk_amount / (entry_price - stop_price)


def cap_by_max_position(
    quantity: Decimal,
    entry_price: Decimal,
    equity: Decimal,
    limits: RiskLimits,
    existing_position_value: Decimal = Decimal(0),
) -> Decimal:
    """Cap so existing exposure plus this order stays within max_position_pct; may be 0.

    The cap is per symbol, so existing_position_value includes orders already approved
    earlier in the batch.
    """
    max_value = equity * limits.max_position_pct - existing_position_value
    if max_value <= 0:
        return Decimal(0)
    if quantity * entry_price <= max_value:
        return quantity
    return max_value / entry_price


def deployable_cash(portfolio: PortfolioState, limits: RiskLimits) -> Decimal:
    """Cash available for new positions after the reserve; never negative."""
    reserve = portfolio.equity * limits.cash_reserve_pct
    return max(Decimal(0), portfolio.cash - reserve)


def cap_by_available_cash(quantity: Decimal, entry_price: Decimal, available: Decimal) -> Decimal:
    """Cap a quantity so its notional fits within `available`, which is net of the reserve."""
    if quantity * entry_price <= available:
        return quantity
    return available / entry_price


def sized_quantity(
    portfolio: PortfolioState,
    entry_price: Decimal,
    stop_price: Decimal,
    limits: RiskLimits,
    existing_position_value: Decimal = Decimal(0),
    available_cash: Decimal | None = None,
) -> Decimal:
    """Risk-based size, then both caps; may return 0, ValueError on a bad stop or entry.

    available_cash is the batch's running remainder, defaulting to deployable cash. The
    risk budget is portfolio.risk_equity when set, else equity.
    """
    if available_cash is None:
        available_cash = deployable_cash(portfolio, limits)
    risk_equity = portfolio.risk_equity if portfolio.risk_equity is not None else portfolio.equity
    qty = risk_based_quantity(risk_equity, entry_price, stop_price, limits)
    qty = cap_by_max_position(qty, entry_price, portfolio.equity, limits, existing_position_value)
    qty = cap_by_available_cash(qty, entry_price, max(Decimal(0), available_cash))
    return max(Decimal(0), qty)
