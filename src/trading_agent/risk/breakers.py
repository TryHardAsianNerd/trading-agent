"""Layers 3 and 4: circuit breakers and the max-drawdown condition.

Pure functions over PortfolioState, so they are testable without Redis; the gate turns
a trip into rejections and a cooldown. Losses are measured against the period's
starting equity.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.models import PortfolioState, RejectReason


def _loss_pct(start_equity: Decimal, equity_now: Decimal) -> Decimal:
    """Fractional loss since start_equity (negative on a gain); 0 without a positive baseline."""
    if start_equity <= 0:
        return Decimal(0)
    return (start_equity - equity_now) / start_equity


def daily_loss_pct(p: PortfolioState) -> Decimal:
    """Fractional loss since day start."""
    return _loss_pct(p.day_start_equity, p.equity)


def weekly_loss_pct(p: PortfolioState) -> Decimal:
    """Fractional loss since week start."""
    return _loss_pct(p.week_start_equity, p.equity)


def drawdown_pct(p: PortfolioState) -> Decimal:
    """Fractional drawdown from peak equity."""
    return _loss_pct(p.peak_equity, p.equity)


def tripped_breaker(p: PortfolioState, limits: RiskLimits) -> RejectReason | None:
    """The most severe tripped condition (drawdown, daily, weekly, consecutive), or None.

    Drawdown outranks the Layer 3 breakers: the gate turns it into a kill-switch
    activation, not just a rejection.
    """
    if drawdown_pct(p) >= limits.max_drawdown_pct:
        return RejectReason.MAX_DRAWDOWN_EXCEEDED
    if daily_loss_pct(p) >= limits.daily_loss_halt_pct:
        return RejectReason.DAILY_LOSS_BREAKER
    if weekly_loss_pct(p) >= limits.weekly_loss_pause_pct:
        return RejectReason.WEEKLY_LOSS_BREAKER
    if p.consecutive_losses >= limits.max_consecutive_losses:
        return RejectReason.CONSECUTIVE_LOSSES_BREAKER
    return None


def seconds_until_week_end(now: datetime | None = None) -> int:
    """Whole seconds from `now` (default: the current time) to next Monday 00:00 UTC.

    The weekly breaker pauses for the rest of the week, not just the 1-hour minimum.
    """
    now = now or datetime.now(UTC)
    days_ahead = 7 - now.weekday()  # Monday is 0
    next_monday = (now + timedelta(days=days_ahead)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int((next_monday - now).total_seconds())
