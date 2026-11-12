"""The paper ledger's pure rules: splits, the schedule, offsets, anchor growth, the start."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from trading_agent.brokers.base import AccountSummary, BrokerError, BrokerPosition
from trading_agent.portfolio import ledger
from trading_agent.storage.models import CashFlow

T0 = datetime(2026, 10, 1, 14, 0, tzinfo=UTC)
D = Decimal


def _row(kind, venue, amount, at=T0, equity_before="0", scheduled_for=None) -> CashFlow:
    """An unsaved CashFlow row."""
    return CashFlow(
        ts=at,
        trading_mode="paper",
        asset_class=venue,
        kind=kind,
        amount=D(amount),
        equity_before=D(equity_before),
        scheduled_for=scheduled_for,
    )


def _venue(asset_class: str, cash: str, positions=()) -> MagicMock:
    """A fake broker whose cash is also its equity."""
    broker = MagicMock()
    broker.name = asset_class
    broker.asset_class = asset_class
    broker.get_account.return_value = AccountSummary(equity=D(cash), cash=D(cash))
    broker.get_positions.return_value = list(positions)
    return broker


def test_a_deposit_sends_the_crypto_share_to_the_crypto_venue():
    assert ledger.split(D(500), {"stock", "crypto"}) == {"crypto": D(50), "stock": D(450)}
    assert ledger.split(D(500), {"stock"}) == {"stock": D(500)}


def test_deposit_rows_chain_equity_before_so_growth_factors_compose():
    rows = ledger.deposit_rows(D(500), {"stock", "crypto"}, D(1000), T0, T0.date(), "paper")
    assert [(r.asset_class, r.amount, r.equity_before) for r in rows] == [
        ("crypto", D(50), D(1000)),
        ("stock", D(450), D(1050)),
    ]
    assert ledger.growth_after(rows, T0 - timedelta(seconds=1)) == D("1.5")


def test_each_deposit_row_records_the_cores_part_when_booked():
    """$500: the stock row is $450 with the core's $300 (swing gets $150), and the
    crypto row's $50 is all core. A stock-only ledger keeps the same core share."""
    rows = ledger.deposit_rows(D(500), {"stock", "crypto"}, D(0), T0, T0.date(), "paper")
    assert [(r.asset_class, r.amount, r.core_amount) for r in rows] == [
        ("crypto", D(50), D(50)),
        ("stock", D(450), D(300)),
    ]
    (only,) = ledger.deposit_rows(D(500), {"stock"}, D(0), T0, T0.date(), "paper")
    assert (only.amount, only.core_amount) == (D(500), D(300))


def test_due_dates_step_every_14_days_and_skip_booked_ones():
    flows = [
        _row("baseline", "stock", "-100000"),
        _row("deposit", "stock", "1000", scheduled_for=T0.date()),
        _row("deposit", "stock", "500", scheduled_for=date(2026, 10, 15)),
    ]
    assert ledger.due_dates(flows, date(2026, 10, 14)) == []
    assert ledger.due_dates(flows, date(2026, 11, 12)) == [date(2026, 10, 29), date(2026, 11, 12)]


def test_offsets_sum_each_venues_rows():
    flows = [
        _row("baseline", "stock", "-100000"),
        _row("baseline", "crypto", "-10000"),
        _row("deposit", "stock", "900"),
        _row("deposit", "crypto", "100"),
    ]
    assert ledger.offsets(flows) == {"stock": D(-99100), "crypto": D(-9900)}


def test_growth_counts_only_later_deposits_into_a_funded_account():
    flows = [
        _row("baseline", "stock", "-100000"),
        _row("deposit", "stock", "1000", T0, "0"),  # into an empty account: no anchor precedes it
        _row("deposit", "stock", "500", T0 + timedelta(days=14), "900"),
        _row("deposit", "stock", "500", T0 + timedelta(days=28), "1400"),
    ]
    assert ledger.growth_after(flows, T0 + timedelta(days=14)) == D(1900) / D(1400)
    assert ledger.growth_after(flows, T0) == D(1400) / D(900) * D(1900) / D(1400)
    assert ledger.growth_after(flows, T0 + timedelta(days=30)) == 1


def test_starting_cancels_each_venues_cash_and_deposits_the_first_1000():
    rows = ledger.start_ledger([_venue("stock", "100000"), _venue("crypto", "10000")], T0)
    assert [(r.kind, r.asset_class, r.amount, r.core_amount) for r in rows] == [
        ("baseline", "stock", D(-100000), D(0)),
        ("baseline", "crypto", D(-10000), D(0)),
        ("deposit", "crypto", D(100), D(100)),
        ("deposit", "stock", D(900), D(600)),
    ]
    assert ledger.offsets(rows) == {"stock": D(-99100), "crypto": D(-9900)}
    assert rows[-1].scheduled_for == T0.date()


def test_starting_refuses_a_venue_that_holds_positions():
    held = BrokerPosition(symbol="AAPL", asset_class="stock", quantity=D(1), avg_entry_price=D(1))
    with pytest.raises(ValueError, match="AAPL"):
        ledger.start_ledger([_venue("stock", "100000", [held])], T0)
    with pytest.raises(ValueError, match="no venues"):
        ledger.start_ledger([], T0)


def test_the_cli_refuses_live_mode(monkeypatch):
    monkeypatch.setattr(ledger, "load_settings", lambda: MagicMock(is_paper=False))
    monkeypatch.setattr(ledger, "configure_logging", lambda: None)
    with pytest.raises(SystemExit, match="paper mode only"):
        ledger.main(["start"])


# --- start --close-positions (the paper cutover, D-046) ---------------------------------


def _held(symbol: str, qty: str, asset_class: str = "stock") -> BrokerPosition:
    """A held BrokerPosition."""
    return BrokerPosition(
        symbol=symbol, asset_class=asset_class, quantity=D(qty), avg_entry_price=D(1)
    )


def _stub_cli(monkeypatch) -> None:
    """Stub the CLI's settings, logging, database and venues (paper mode, no venues)."""
    from trading_agent.graph import deps

    monkeypatch.setattr(ledger, "load_settings", lambda: MagicMock(is_paper=True))
    monkeypatch.setattr(ledger, "configure_logging", lambda: None)
    monkeypatch.setattr(ledger, "make_engine", lambda url: None)
    monkeypatch.setattr(ledger, "make_session_factory", lambda engine: MagicMock())
    monkeypatch.setattr(deps, "make_deps", lambda settings, factory: MagicMock(brokers=[]))


def test_closing_cancels_orders_sells_everything_and_waits_until_flat():
    stock = _venue("stock", "90000", [_held("AAPL", "32"), _held("NVDA", "24")])
    crypto = _venue("crypto", "10000", [_held("BTC-USD", "0.1", "crypto")])
    # Held while the sells go out, flat at the first check after them.
    stock.get_positions.side_effect = [stock.get_positions.return_value, []]
    crypto.get_positions.side_effect = [crypto.get_positions.return_value, []]
    naps = []

    sold = ledger.close_positions([stock, crypto], T0, sleep=naps.append, clock=lambda: T0)

    assert sold == ["stock:AAPL", "stock:NVDA", "crypto:BTC-USD"]
    stock.cancel_all_orders.assert_called_once()
    crypto.cancel_all_orders.assert_called_once()
    orders = [c.args for c in stock.submit_order.call_args_list]
    assert [(o.symbol, o.side, o.quantity) for o, _ in orders] == [
        ("AAPL", "sell", D(32)),
        ("NVDA", "sell", D(24)),
    ]
    assert orders[0][1] == f"cutover-{T0:%Y%m%d}:AAPL"  # replay-safe idempotency key
    assert naps == []  # flat on the first check after the sells


def test_closing_gives_up_with_what_is_still_held():
    """Outside the NYSE session a stock sell never fills: stop at the timeout."""
    stock = _venue("stock", "90000", [_held("AAPL", "32")])
    times = iter([T0, T0 + ledger.CLOSE_TIMEOUT])
    with pytest.raises(ValueError, match="stock:AAPL"):
        ledger.close_positions([stock], T0, sleep=lambda s: None, clock=lambda: next(times))


def test_closing_retries_a_sell_refused_while_the_cancel_is_still_pending():
    """Alpaca cancels in the background: a stop in pending_cancel still holds its
    shares, so the first sell can be refused ("insufficient qty available")."""
    stock = _venue("stock", "90000", [_held("AAPL", "32")])
    held = stock.get_positions.return_value
    stock.get_positions.side_effect = [held, held, []]  # listed, still held, flat
    stock.submit_order.side_effect = [BrokerError("insufficient qty available"), MagicMock()]
    naps = []

    sold = ledger.close_positions([stock], T0, sleep=naps.append, clock=lambda: T0)

    assert sold == ["stock:AAPL"]
    assert stock.submit_order.call_count == 2
    assert naps == [ledger.CLOSE_POLL_SECONDS]  # one wait between the two tries


def test_closing_gives_up_on_a_sell_still_refused_at_the_timeout():
    stock = _venue("stock", "90000", [_held("AAPL", "32")])
    stock.submit_order.side_effect = BrokerError("insufficient qty available")
    times = iter([T0, T0 + ledger.CLOSE_TIMEOUT])
    with pytest.raises(ValueError, match=r"stock:AAPL.*insufficient qty"):
        ledger.close_positions([stock], T0, sleep=lambda s: None, clock=lambda: next(times))


def test_the_cli_turns_a_venue_failure_into_a_message(monkeypatch):
    def refuse(brokers, now):
        """Fail like a venue would."""
        raise BrokerError("Alpaca cancel_orders failed")

    monkeypatch.setattr(ledger, "close_positions", refuse)
    _stub_cli(monkeypatch)
    with pytest.raises(SystemExit, match="Alpaca cancel_orders failed"):
        ledger.main(["start", "--close-positions"])


def test_the_cli_closes_positions_before_starting(monkeypatch):
    calls = []
    monkeypatch.setattr(ledger, "close_positions", lambda b, now: calls.append("close") or [])
    monkeypatch.setattr(ledger, "start_ledger", lambda b, now: calls.append("start") or [])
    _stub_cli(monkeypatch)

    ledger.main(["start", "--close-positions"])

    assert calls == ["close", "start"]
