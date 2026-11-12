"""Layer 2 tests: fixed-fractional sizing and the caps (position %, cash reserve)."""

from decimal import Decimal

import pytest

from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.models import PortfolioState
from trading_agent.risk.sizing import (
    cap_by_available_cash,
    cap_by_max_position,
    deployable_cash,
    risk_based_quantity,
    sized_quantity,
)

LIMITS = RiskLimits()


def _portfolio(equity: str, cash: str) -> PortfolioState:
    e = Decimal(equity)
    return PortfolioState(
        equity=e,
        cash=Decimal(cash),
        peak_equity=e,
        day_start_equity=e,
        week_start_equity=e,
        consecutive_losses=0,
    )


def test_risk_based_quantity_exact():
    # 1% of 100k = 1000 at risk; stop distance 4 → 250 units.
    qty = risk_based_quantity(Decimal(100_000), Decimal(100), Decimal(96), LIMITS)
    assert qty == Decimal(250)


def test_stop_at_or_above_entry_rejected():
    with pytest.raises(ValueError, match="below entry"):
        risk_based_quantity(Decimal(100_000), Decimal(100), Decimal(100), LIMITS)


def test_nonpositive_stop_rejected():
    with pytest.raises(ValueError, match="positive"):
        risk_based_quantity(Decimal(100_000), Decimal(100), Decimal("-2"), LIMITS)


def test_cap_by_max_position_binds():
    # 250 x 100 = 25,000 value, cap is 10% of 100k = 10,000 → 100 units.
    assert cap_by_max_position(Decimal(250), Decimal(100), Decimal(100_000), LIMITS) == Decimal(100)


def test_cap_by_max_position_no_bind():
    assert cap_by_max_position(Decimal(10), Decimal(100), Decimal(100_000), LIMITS) == Decimal(10)


def test_deployable_cash_reserves_20_pct_of_equity():
    p = _portfolio("100000", "30000")
    assert deployable_cash(p, LIMITS) == Decimal(10_000)  # 30k cash - 20k reserve


def test_deployable_cash_never_negative():
    p = _portfolio("100000", "5000")  # cash below the reserve already
    assert deployable_cash(p, LIMITS) == Decimal(0)


def test_cap_by_available_cash_binds():
    assert cap_by_available_cash(Decimal(50), Decimal(100), Decimal(2000)) == Decimal(20)


def test_cap_by_max_position_counts_existing_exposure():
    # 10% cap = 10,000; existing 3,000 leaves 7,000 headroom -> 70 units at 100.
    qty = cap_by_max_position(
        Decimal(250), Decimal(100), Decimal(100_000), LIMITS, existing_position_value=Decimal(3000)
    )
    assert qty == Decimal(70)


def test_cap_by_max_position_zero_headroom():
    qty = cap_by_max_position(
        Decimal(10), Decimal(100), Decimal(100_000), LIMITS, existing_position_value=Decimal(10000)
    )
    assert qty == Decimal(0)


def test_sized_quantity_full_pipeline():
    # risk size 250 → 10% position cap → 100 → deployable 80k does not bind.
    p = _portfolio("100000", "100000")
    assert sized_quantity(p, Decimal(100), Decimal(96), LIMITS) == Decimal(100)


def test_sized_quantity_zero_when_reserve_exhausted():
    p = _portfolio("100000", "20000")  # exactly at the reserve floor
    assert sized_quantity(p, Decimal(100), Decimal(96), LIMITS) == Decimal(0)


def test_sized_quantity_supports_fractional_result():
    # Crypto-scale: tiny deployable cash yields a fractional quantity, not zero.
    p = _portfolio("1000", "201")  # deployable = 1
    qty = sized_quantity(p, Decimal(100), Decimal(96), LIMITS)
    assert qty == Decimal("0.01")


def test_hitting_stop_loses_at_most_risk_pct():
    """Property: for any binding combination, loss at stop <= 1% of equity."""
    p = _portfolio("100000", "100000")
    for entry, stop in [
        (Decimal(100), Decimal(96)),
        (Decimal(50), Decimal(49)),
        (Decimal(3), Decimal("2.9")),
        (Decimal(70000), Decimal(65000)),
    ]:
        qty = sized_quantity(p, entry, stop, LIMITS)
        loss_at_stop = qty * (entry - stop)
        assert loss_at_stop <= p.equity * LIMITS.risk_per_trade_pct


def test_risk_per_trade_is_a_share_of_risk_equity_when_set():
    """The swing sleeve's view (D-040 item 5): caps and reserve on the sleeve's
    equity, the 1% risk budget on the whole account's."""
    sleeve = _portfolio("1000", "1000").model_copy(update={"risk_equity": Decimal(10_000)})
    limits = RiskLimits(max_position_pct=Decimal("0.25"))
    # Risk: 1% of 10,000 = 100 over a 10 stop distance = 10 units; cap 25% of 1,000 = 2.5 units.
    assert sized_quantity(sleeve, Decimal(100), Decimal(90), limits) == Decimal("2.5")
    loose = limits.__class__(max_position_pct=Decimal(1))
    # Without the cap the account-sized risk budget binds (the reserve leaves 800 = 8 units).
    assert sized_quantity(sleeve, Decimal(100), Decimal(90), loose) == Decimal(8)
    plain = _portfolio("1000", "1000")
    assert sized_quantity(plain, Decimal(100), Decimal(90), loose) == Decimal(1)  # 1% of 1,000
