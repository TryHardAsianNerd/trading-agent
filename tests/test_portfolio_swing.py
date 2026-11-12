"""The swing sleeve's view for the gate (D-040 items 4-6, D-042)."""

from decimal import Decimal as D

from trading_agent.portfolio import swing
from trading_agent.risk.breakers import daily_loss_pct, drawdown_pct
from trading_agent.risk.models import PortfolioState, Position


def _account(stock_cash: str, positions=(), equity: str = "1000") -> PortfolioState:
    """The whole account on ledger terms; positions are (symbol, qty, avg) tuples."""
    return PortfolioState(
        equity=D(equity),
        cash=D(stock_cash),
        peak_equity=D(equity),
        day_start_equity=D(equity),
        week_start_equity=D(equity),
        consecutive_losses=0,
        positions={
            s: Position(symbol=s, asset_class="stock", quantity=D(q), avg_entry_price=D(a))
            for s, q, a in positions
        },
        untracked_positions={"VOO": D(1), "AAPL": D(2)},
        venue_cash={"stock": D(stock_cash), "crypto": D(5)},
    )


def test_swing_equity_is_stock_cash_after_the_core_plus_non_core_positions():
    account = _account("900", [("AAPL", "2", "100"), ("VOO", "3", "100")])
    cash, equity = swing.equity(account, {"AAPL": D(110)}, D(600))
    assert (cash, equity) == (D(300), D(300) + D(220))  # VOO is core; AAPL at market
    # A position without a mark counts at cost; cash below the core's claim stays negative
    # (clamping it would make part of the next deposit read as a loss).
    cash, equity = swing.equity(account, {}, D(2000))
    assert (cash, equity) == (D(-1100), D(-900))


def test_the_swing_sleeve_gets_30_percent_of_every_deposit():
    """D-048 (D-040 restored): the stock venue gets 90% of each deposit and the
    core claims 60%, so the swing sleeve's cash is the other 30%."""
    from datetime import UTC, date, datetime

    from trading_agent.portfolio import core, ledger

    at = datetime(2026, 10, 2, 14, tzinfo=UTC)
    flows = []
    for n, amount in enumerate((D(1000), D(500), D(500))):
        flows += ledger.deposit_rows(
            amount, {"stock", "crypto"}, D(0), at, date(2026, 10, 2 + n), "paper"
        )
    stock_cash = sum((f.amount for f in flows if f.asset_class == "stock"), D(0))
    claim = core.stock_core_cash(flows, [], D(0))
    assert (stock_cash, claim) == (D(1800), D(1200))
    assert swing.equity(_account(str(stock_cash), equity="2000"), {}, claim) == (D(600), D(600))


def test_the_index_starts_at_1_and_ignores_new_money():
    assert swing.next_index(None, D(300), D(0)) == D(1)
    # 300 -> 450 with a 150 deposit: no return at all.
    assert swing.next_index((D(300), D(1)), D(450), D(150)) == D(1)
    # 300 -> 270 with nothing deposited: -10%.
    assert swing.next_index((D(300), D(2)), D(270), D(0)) == D("1.8")
    # An empty sleeve has no return to measure.
    assert swing.next_index((D(0), D("0.9")), D(150), D(150)) == D("0.9")


def test_a_deposit_neither_hides_nor_dilutes_a_days_loss():
    """Day starts at index 1; the sleeve loses 10% (index 0.9), then a deposit
    lands. The breaker must still read -10%, not a smaller or no loss."""
    index = swing.next_index((D(300), D(1)), D(420), D(150))  # 270 + 150 deposited
    assert index == D("0.9")
    view = swing.view(_account("420"), D(420), D(420), index, (D(1), D(1), D(1)), 2)
    assert round(daily_loss_pct(view), 12) == D("0.1")
    assert round(drawdown_pct(view), 12) == D("0.1")


def test_the_view_is_the_sleeve_with_the_accounts_equity_for_risk():
    account = _account("900", [("AAPL", "2", "100"), ("VOO", "3", "100")], equity="1500")
    view = swing.view(account, D(300), D(500), D(1), (None, None, None), 3)
    assert (view.equity, view.cash, view.risk_equity) == (D(500), D(300), D(1500))
    assert set(view.positions) == {"AAPL"} and set(view.untracked_positions) == {"AAPL"}
    assert view.venue_cash == {"stock": D(300), "crypto": D(0)}  # crypto cash is all core
    assert view.consecutive_losses == 3
    # No anchor snapshot yet: no loss, and the peak is today's equity.
    assert (view.day_start_equity, view.week_start_equity, view.peak_equity) == (D(500),) * 3


def test_a_wiped_out_sleeve_reads_as_a_total_loss_and_can_still_recover():
    index = swing.next_index((D(300), D(1)), D(0), D(0))
    assert index == swing.INDEX_FLOOR  # never 0: 0 x any later return would stay 0
    view = swing.view(_account("0"), D(0), D(0), index, (D(1), D(1), D(1)), 0)
    assert daily_loss_pct(view) == D(1) and drawdown_pct(view) == D(1)
    # A later deposit and gain move the index again.
    later = swing.next_index((D(0), index), D(150), D(150))
    assert swing.next_index((D(150), later), D(165), D(0)) > later


def test_swing_limits_cap_a_position_at_25_percent_and_keep_the_20_percent_reserve():
    assert swing.SWING_LIMITS.max_position_pct == D("0.25")
    assert swing.SWING_LIMITS.cash_reserve_pct == D("0.20")
    assert swing.SWING_LIMITS.risk_per_trade_pct == D("0.01")


def test_a_sleeve_never_funded_reads_no_loss():
    """Empty from the first snapshot (index still 1): not a wipe-out, no kill switch."""
    view = swing.view(_account("0"), D(0), D(0), D(1), (D(1), D(1), D(1)), 0)
    assert daily_loss_pct(view) == D(0) and drawdown_pct(view) == D(0)


def test_unexplained_stock_cash_is_what_no_deposit_or_fill_accounts_for():
    """A $1.03 venue fee the book never saw reads as -1.03; a booked fill does not."""
    from datetime import UTC, datetime, timedelta

    from trading_agent.storage.models import CashFlow, Trade

    start = datetime(2026, 10, 2, 13, tzinfo=UTC)
    flows = [
        CashFlow(ts=start, asset_class="stock", kind="baseline", amount=D(-100000)),
        CashFlow(ts=start, asset_class="stock", kind="deposit", amount=D(900)),
        CashFlow(ts=start, asset_class="crypto", kind="deposit", amount=D(100)),
    ]
    buy = Trade(
        filled_at=start + timedelta(hours=1),
        side="buy",
        quantity=D(2),
        fill_price=D(100),
        fees=D(0),
    )
    early = Trade(
        filled_at=start - timedelta(days=1), side="sell", quantity=D(5), fill_price=D(1), fees=D(0)
    )
    assert swing.unexplained(D(700), flows, [buy, early]) == 0
    assert swing.unexplained(D("698.97"), flows, [buy]) == D("-1.03")
