"""Layer 3+4 tests: circuit breakers and drawdown, with boundary cases."""

from decimal import Decimal

from trading_agent.risk.breakers import (
    daily_loss_pct,
    drawdown_pct,
    tripped_breaker,
    weekly_loss_pct,
)
from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.models import PortfolioState, RejectReason

LIMITS = RiskLimits()


def _p(
    equity: str = "100000",
    peak: str = "100000",
    day_start: str = "100000",
    week_start: str = "100000",
    losses: int = 0,
) -> PortfolioState:
    return PortfolioState(
        equity=Decimal(equity),
        cash=Decimal(equity),
        peak_equity=Decimal(peak),
        day_start_equity=Decimal(day_start),
        week_start_equity=Decimal(week_start),
        consecutive_losses=losses,
    )


def test_no_breaker_when_healthy():
    assert tripped_breaker(_p(), LIMITS) is None


def test_daily_loss_below_threshold_passes():
    assert tripped_breaker(_p(equity="97001"), LIMITS) is None  # 2.999% loss


def test_daily_loss_at_threshold_trips():
    p = _p(equity="97000")  # exactly 3%
    assert daily_loss_pct(p) == Decimal("0.03")
    assert tripped_breaker(p, LIMITS) is RejectReason.DAILY_LOSS_BREAKER


def test_weekly_loss_trips_when_daily_ok():
    # Down 5% on the week but flat today.
    p = _p(equity="95000", day_start="95000", week_start="100000")
    assert weekly_loss_pct(p) == Decimal("0.05")
    assert tripped_breaker(p, LIMITS) is RejectReason.WEEKLY_LOSS_BREAKER


def test_consecutive_losses_boundary():
    assert tripped_breaker(_p(losses=4), LIMITS) is None
    assert tripped_breaker(_p(losses=5), LIMITS) is RejectReason.CONSECUTIVE_LOSSES_BREAKER


def test_drawdown_trips_at_15_pct_from_peak():
    p = _p(equity="85000", peak="100000", day_start="85000", week_start="85000")
    assert drawdown_pct(p) == Decimal("0.15")
    assert tripped_breaker(p, LIMITS) is RejectReason.MAX_DRAWDOWN_EXCEEDED


def test_drawdown_outranks_other_breakers():
    # Everything is on fire; drawdown must win (it drives kill-switch activation).
    p = _p(equity="80000", peak="100000", day_start="90000", week_start="95000", losses=9)
    assert tripped_breaker(p, LIMITS) is RejectReason.MAX_DRAWDOWN_EXCEEDED


def test_gains_never_trip():
    p = _p(equity="120000", peak="120000")
    assert tripped_breaker(p, LIMITS) is None
    assert daily_loss_pct(p) < 0  # a gain reads as negative loss


def test_zero_start_equity_is_safe():
    p = _p(equity="0", peak="0", day_start="0", week_start="0")
    assert tripped_breaker(p, LIMITS) is None  # no division blowup on empty account
