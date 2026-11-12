"""Trade executor node tests.

Hermetic tests cover the pure P&L helper. Node tests run against the real
compose TimescaleDB (skipped when unreachable) with an in-memory venue fake
that behaves like a real broker for the executor's purposes: synchronous or
asynchronous fills, venue-side stop fires, client_order_id replays, and
typed failures. Rows use broker "test", trading modes "t-*", cycle ids
"test-*" and are cleaned up by the module fixture.
"""

import contextlib
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from tests.fakes import BrokenRedis, FakeRedis

from trading_agent.brokers.base import (
    BrokerError,
    BrokerPosition,
    SubmittedOrder,
    UnfillableQuantityError,
    UnknownOrderError,
    UnprotectedPositionError,
    VenueOrder,
    WrongVenueError,
)
from trading_agent.data.models import Candle
from trading_agent.graph.nodes import trade_executor as executor_mod
from trading_agent.graph.nodes.trade_executor import execute, realized_pnl
from trading_agent.graph.state import ApprovedOrder, TradeProposal
from trading_agent.portfolio.tracker import average_cost_walk
from trading_agent.risk.killswitch import COOLDOWN_KEY, KILL_SWITCH_KEY, KillSwitch
from trading_agent.storage.db import init_db, make_engine, make_session_factory
from trading_agent.storage.models import Decision, Order, Trade
from trading_agent.storage.repos import OrderRepository, TradeRepository


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    """The one-retry delay must not slow the suite down."""
    monkeypatch.setattr("trading_agent.graph.nodes.trade_executor.time.sleep", lambda _s: None)


# ------------------------------------------------------------ hermetic


def _trade(side, qty, price, days_ago=1):
    return Trade(
        side=side,
        quantity=Decimal(qty),
        fill_price=Decimal(price),
        filled_at=datetime.now(UTC) - timedelta(days=days_ago),
    )


def test_realized_pnl_uses_average_cost_basis_net_of_fees():
    fills = [_trade("buy", "10", "100", 3), _trade("buy", "10", "120", 2)]  # avg 110
    assert realized_pnl(fills, Decimal("115"), Decimal("20"), Decimal("1")) == Decimal("99")


def test_realized_pnl_is_none_without_an_open_basis():
    """Unknown basis must never count as a loss or a win for the streak."""
    assert realized_pnl([], Decimal("90"), Decimal("1"), Decimal(0)) is None
    flat = [_trade("buy", "5", "100", 3), _trade("sell", "5", "90", 2)]
    assert realized_pnl(flat, Decimal("90"), Decimal("1"), Decimal(0)) is None


# ------------------------------------------------------------- fixtures


def _engine_or_skip():
    from trading_agent.config import Settings

    settings = Settings(_env_file=None)
    engine = make_engine(settings.database_url, connect_args={"connect_timeout": 3})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("TimescaleDB not reachable — run: docker compose up -d")
    return engine


@pytest.fixture(scope="module")
def engine():
    eng = _engine_or_skip()
    init_db(eng)
    yield eng
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM trades WHERE broker LIKE 'test%'"))
        conn.execute(text("DELETE FROM orders WHERE broker LIKE 'test%'"))
        conn.execute(text("DELETE FROM decisions WHERE cycle_id LIKE 'test-%'"))


@pytest.fixture()
def session_factory(engine):
    return make_session_factory(engine)


class FakeVenue:
    """In-memory venue: orders keyed by venue id, fills controlled by tests.

    sync=True fills entries immediately (PaperBroker-like); sync=False leaves
    them accepted until the test calls fill() (Alpaca-like). Buys with a
    stop get a protective order that stays open until fire_stop().
    """

    def __init__(self, asset_class="stock", sync=True, fill_price="100", name="test"):
        self.name = name
        self.asset_class = asset_class
        self.sync = sync
        self.fill_price = Decimal(fill_price)
        self.orders: dict[str, VenueOrder] = {}
        self.quantities: dict[str, Decimal] = {}
        self.submitted: list[tuple[ApprovedOrder, str | None]] = []
        self.by_client_id: dict[str, SubmittedOrder] = {}
        self.fail_with: Exception | None = None
        self.fail_times = 0
        self.lookup_fails = False
        self.cancel_calls = 0
        self.cancel_with: Exception | None = None
        self._count = 0
        self._prefix = uuid.uuid4().hex[:6]  # venue ids are globally unique

    def oid(self, n):
        """Venue id of this instance's n-th accepted order."""
        return f"v-{self._prefix}-{n}"

    def submit_order(self, order, client_order_id=None):
        if client_order_id in self.by_client_id:
            return self.by_client_id[client_order_id]  # venue-side replay
        self.submitted.append((order, client_order_id))
        if self.fail_with is not None and self.fail_times > 0:
            self.fail_times -= 1
            raise self.fail_with
        self._count += 1
        oid = self.oid(self._count)
        self.orders[oid] = VenueOrder(status="accepted", terminal=False)
        self.quantities[oid] = order.quantity
        if self.sync:
            self.fill(oid)
        stop_id = None
        if order.side == "buy" and order.stop_loss_price is not None:
            stop_id = f"{oid}-stop"
            self.orders[stop_id] = VenueOrder(status="held", terminal=False)
            self.quantities[stop_id] = order.quantity
        submitted = SubmittedOrder(
            broker_order_id=oid,
            status=self.orders[oid].status,
            submitted_quantity=order.quantity,
            client_order_id=client_order_id,
            protective_order_id=stop_id,
        )
        if client_order_id:
            self.by_client_id[client_order_id] = submitted
        return submitted

    def fill(self, oid, price=None):
        self.orders[oid] = self.orders[oid].model_copy(
            update={
                "status": "filled",
                "terminal": True,
                "filled_quantity": self.quantities[oid],
                "fill_price": price if price is not None else self.fill_price,
                "filled_at": datetime.now(UTC),
            }
        )

    def fire_stop(self, oid, price):
        self.fill(f"{oid}-stop", price=Decimal(price))

    def cancel(self, oid):
        self.orders[oid] = self.orders[oid].model_copy(
            update={"status": "canceled", "terminal": True}
        )

    def get_order(self, broker_order_id):
        if self.lookup_fails:
            raise BrokerError("venue lookup down")
        try:
            return self.orders[broker_order_id]
        except KeyError as e:
            raise UnknownOrderError(f"unknown order {broker_order_id}") from e

    def cancel_all_orders(self):
        self.cancel_calls += 1
        if self.cancel_with is not None:
            raise self.cancel_with
        open_ids = [oid for oid, o in self.orders.items() if not o.terminal]
        for oid in open_ids:
            self.cancel(oid)
        return len(open_ids)

    def get_positions(self):
        return []

    def get_account(self):  # pragma: no cover - the executor never calls it
        raise AssertionError("executor must not read the account")


def _deps(session_factory, *brokers, redis=None):
    """Minimal CycleDeps: only what the executor touches."""
    mode = f"t-{uuid.uuid4().hex[:6]}"
    return SimpleNamespace(
        session_factory=session_factory,
        brokers=list(brokers),
        kill_switch=KillSwitch(redis or FakeRedis()),
        tracker=SimpleNamespace(trading_mode=mode),
    ), mode


def _order(symbol, side="buy", quantity="10", stop="95", asset_class="stock", proposal_class=None):
    return ApprovedOrder(
        symbol=symbol,
        asset_class=asset_class,
        side=side,
        quantity=Decimal(quantity),
        stop_loss_price=Decimal(stop) if stop is not None else None,
        source_proposal=TradeProposal(
            symbol=symbol,
            asset_class=proposal_class or asset_class,
            action=side,
            confidence=0.9,
            reasoning="t",
        ),
    )


def _symbol():
    return f"T-{uuid.uuid4().hex[:6].upper()}"


def _cycle():
    return f"test-{uuid.uuid4().hex[:8]}"


def _fills(session_factory, symbol, mode):
    with session_factory() as session:
        return TradeRepository(session).fills_for_symbol(symbol, mode)


def _orders(session_factory, symbol):
    with session_factory() as session:
        return (
            session.query(Order)
            .filter(Order.symbol == symbol, Order.broker == "test")
            .order_by(Order.id)
            .all()
        )


def _executor_row(session_factory, cycle_id):
    with session_factory() as session:
        return (
            session.query(Decision)
            .filter(Decision.cycle_id == cycle_id, Decision.node == "trade_executor")
            .one()
        )


# ----------------------------------------------------------- node tests


def test_buy_is_routed_booked_and_audited(session_factory):
    venue = FakeVenue()
    deps, mode = _deps(session_factory, venue)
    symbol, cycle_id = _symbol(), _cycle()

    out = execute({"cycle_id": cycle_id, "approved_orders": [_order(symbol)]}, deps=deps)

    (result,) = out["execution_results"]
    assert result["status"] == "filled"
    assert result["client_order_id"] == f"{cycle_id}:{symbol}"
    assert result["broker_order_id"] == venue.oid(1)
    assert result["protective_order_id"] == venue.oid(1) + "-stop"
    entry, stop = _orders(session_factory, symbol)
    assert (entry.order_type, entry.status, entry.trading_mode) == ("market", "filled", mode)
    assert (stop.order_type, stop.side, stop.status) == ("stop", "sell", "held")
    assert stop.stop_loss_price == Decimal("95")
    (fill,) = _fills(session_factory, symbol, mode)
    assert fill.order_id == entry.id and fill.fill_price == Decimal("100")
    assert fill.realized_pnl is None
    row = _executor_row(session_factory, cycle_id)
    assert row.detail["halted"] is False
    assert row.detail["fills_booked"][0]["broker_order_id"] == venue.oid(1)


def test_routing_uses_the_gate_asset_class_never_the_proposal(session_factory):
    """The LLM's asset_class must not steer venue routing."""
    stocks, crypto = FakeVenue("stock"), FakeVenue("crypto")
    deps, _ = _deps(session_factory, crypto, stocks)
    order = _order(_symbol(), asset_class="stock", proposal_class="crypto")

    execute({"cycle_id": _cycle(), "approved_orders": [order]}, deps=deps)

    assert len(stocks.submitted) == 1 and crypto.submitted == []


def test_no_broker_for_asset_class_is_audited_not_crashed(session_factory):
    deps, _ = _deps(session_factory, FakeVenue("crypto"))
    out = execute({"cycle_id": _cycle(), "approved_orders": [_order(_symbol())]}, deps=deps)
    assert out["execution_results"][0]["status"] == "no_broker"


def test_halted_submits_nothing_but_still_reconciles(session_factory):
    """Kill switch on: no order (not even a breach close) reaches a venue, yet
    a venue-side fill from earlier is still booked and the cycle is audited.
    Every venue's open orders are cancelled BEFORE the poll, so the cancelled
    stop row is retired in the same cycle and the counts audited."""
    venue, crypto = FakeVenue(sync=False), FakeVenue("crypto", name="test-crypto")
    deps, mode = _deps(session_factory, venue, crypto)
    symbol = _symbol()
    execute({"cycle_id": _cycle(), "approved_orders": [_order(symbol)]}, deps=deps)
    venue.fill(venue.oid(1))  # fills at the venue between cycles
    redis = FakeRedis()
    redis.set(KILL_SWITCH_KEY, "manual_discord")
    deps.kill_switch = KillSwitch(redis)
    cycle_id = _cycle()
    breach = {"symbol": symbol, "asset_class": "stock", "quantity": "10"}

    out = execute(
        {"cycle_id": cycle_id, "approved_orders": [_order(_symbol())], "stop_breaches": [breach]},
        deps=deps,
    )

    assert [r["status"] for r in out["execution_results"]] == ["halted", "halted"]
    assert len(venue.submitted) == 1  # only the first cycle's buy
    assert len(_fills(session_factory, symbol, mode)) == 1  # reconciled under halt
    assert (venue.cancel_calls, crypto.cancel_calls) == (1, 1)
    assert [o.status for o in _orders(session_factory, symbol)] == ["filled", "cancelled"]
    row = _executor_row(session_factory, cycle_id)
    assert (
        row.detail["halted"] is True and row.detail["halt_reason"] == "kill_switch:manual_discord"
    )
    assert row.detail["cancelled"] == {"test": 1, "test-crypto": 0}  # the held stop


def test_cooldown_halt_and_unreachable_redis_never_cancel(session_factory):
    """Only the kill switch cancels: a breaker pause and a Redis outage must
    leave protective stops working at the venue (D-019, D-020)."""
    cooldown = FakeRedis()
    cooldown.set(COOLDOWN_KEY, "daily_loss", ex=3600)
    for redis, reason in ((cooldown, "cooldown:daily_loss"), (BrokenRedis(), "redis_unreachable")):
        venue = FakeVenue()
        deps, _ = _deps(session_factory, venue, redis=redis)
        cycle_id = _cycle()

        out = execute({"cycle_id": cycle_id, "approved_orders": [_order(_symbol())]}, deps=deps)

        assert out["execution_results"][0]["status"] == "halted"
        assert venue.cancel_calls == 0 and venue.submitted == []
        row = _executor_row(session_factory, cycle_id)
        assert row.detail["halt_reason"] == reason and row.detail["cancelled"] is None


def test_kill_switch_cancel_failure_is_audited_by_type_and_never_aborts(session_factory):
    """One venue's cancel failure is recorded as a TYPE name (D-018: no venue
    text in the audit trail), is logged via log.exception (error level pages,
    D-018), and the other venues are still cancelled."""
    stocks, crypto = FakeVenue("stock"), FakeVenue("crypto", name="test-crypto")
    stocks.cancel_with = BrokerError("venue said: token=SECRET rejected")
    redis = FakeRedis()
    redis.set(KILL_SWITCH_KEY, "manual_discord")
    deps, _ = _deps(session_factory, stocks, crypto, redis=redis)
    cycle_id = _cycle()

    out = execute({"cycle_id": cycle_id, "approved_orders": [_order(_symbol())]}, deps=deps)

    assert out["execution_results"][0]["status"] == "halted"
    assert (stocks.cancel_calls, crypto.cancel_calls) == (1, 1)
    row = _executor_row(session_factory, cycle_id)
    assert row.detail["cancelled"] == {"test": "BrokerError", "test-crypto": 0}
    assert "SECRET" not in str(row.detail)


def test_breach_closes_full_position_and_books_realized_pnl(session_factory):
    venue = FakeVenue()
    deps, mode = _deps(session_factory, venue)
    symbol, cycle_id = _symbol(), _cycle()
    execute({"cycle_id": _cycle(), "approved_orders": [_order(symbol, quantity="4")]}, deps=deps)
    venue.fill_price = Decimal("90")
    breach = {"symbol": symbol, "asset_class": "stock", "quantity": "4", "stop_price": "97"}

    out = execute({"cycle_id": cycle_id, "stop_breaches": [breach]}, deps=deps)

    (result,) = out["execution_results"]
    assert result["kind"] == "stop_breach" and result["status"] == "filled"
    assert result["client_order_id"] == f"{cycle_id}:{symbol}:stop"
    close, _ = venue.submitted[-1]
    assert close.side == "sell" and close.quantity == Decimal("4")
    assert close.stop_loss_price is None and close.source_proposal is None
    fills = _fills(session_factory, symbol, mode)
    assert fills[-1].side == "sell" and fills[-1].realized_pnl == Decimal("-40")


def test_resumed_cycle_never_double_buys(session_factory):
    """Re-running the executor for the same cycle (crash + resume) hits the
    venue with the same key: one venue order, one orders row, one fill."""
    venue = FakeVenue()
    deps, mode = _deps(session_factory, venue)
    symbol, cycle_id = _symbol(), _cycle()
    state = {"cycle_id": cycle_id, "approved_orders": [_order(symbol)]}

    first = execute(state, deps=deps)
    second = execute(state, deps=deps)

    assert first["execution_results"][0]["broker_order_id"] == venue.oid(1)
    assert second["execution_results"][0]["broker_order_id"] == venue.oid(1)
    assert len(venue.orders) == 2  # entry + its stop, once
    assert len(_orders(session_factory, symbol)) == 2
    assert len(_fills(session_factory, symbol, mode)) == 1


def test_venue_side_stop_fill_is_reconciled_into_the_book(session_factory):
    """A stop that fires at the venue is booked as a losing close and resets the basis."""
    venue = FakeVenue()
    deps, mode = _deps(session_factory, venue)
    symbol = _symbol()
    execute({"cycle_id": _cycle(), "approved_orders": [_order(symbol, quantity="2")]}, deps=deps)
    first_entry_time = average_cost_walk(_fills(session_factory, symbol, mode))[2]
    venue.fire_stop(venue.oid(1), "80")  # gap through the 95 stop

    # Next cycle: reconcile runs first, then the analyst's re-buy executes.
    cycle_id = _cycle()
    out = execute(
        {"cycle_id": cycle_id, "approved_orders": [_order(symbol, quantity="3")]}, deps=deps
    )

    fills = _fills(session_factory, symbol, mode)
    assert [f.side for f in fills] == ["buy", "sell", "buy"]
    assert fills[1].realized_pnl == Decimal("-40") and fills[1].order_id is not None
    with session_factory() as session:
        since = datetime.now(UTC) - timedelta(days=7)
        assert TradeRepository(session).trailing_loss_streak(mode, since=since) == 1
        still_open = OrderRepository(session).open_orders("test", mode)
    assert [o.order_type for o in still_open] == ["stop"]  # only the re-buy's stop
    qty, _, entry_time = average_cost_walk(fills)
    assert qty == Decimal("3") and entry_time > first_entry_time
    assert out["execution_results"][0]["status"] == "filled"
    booked = _executor_row(session_factory, cycle_id).detail["fills_booked"]
    assert [b["order_type"] for b in booked] == ["stop", "market"]


def test_async_fill_is_booked_on_a_later_cycle_and_cancel_retires_the_row(session_factory):
    venue = FakeVenue(sync=False)
    deps, mode = _deps(session_factory, venue)
    symbol = _symbol()
    execute({"cycle_id": _cycle(), "approved_orders": [_order(symbol)]}, deps=deps)
    assert _fills(session_factory, symbol, mode) == []  # accepted, not filled yet

    venue.fill(venue.oid(1))
    venue.cancel(venue.oid(1) + "-stop")
    execute({"cycle_id": _cycle()}, deps=deps)

    (fill,) = _fills(session_factory, symbol, mode)
    assert fill.quantity == Decimal("10")
    assert [o.status for o in _orders(session_factory, symbol)] == ["filled", "cancelled"]
    execute({"cycle_id": _cycle()}, deps=deps)  # terminal rows are not re-booked
    assert len(_fills(session_factory, symbol, mode)) == 1


def test_permanent_errors_are_never_retried(session_factory):
    for error in (WrongVenueError("wrong venue"), UnfillableQuantityError("rounds to 0")):
        venue = FakeVenue()
        venue.fail_with, venue.fail_times = error, 5
        deps, _ = _deps(session_factory, venue)
        out = execute({"cycle_id": _cycle(), "approved_orders": [_order(_symbol())]}, deps=deps)
        assert out["execution_results"][0]["status"] == "rejected_permanent"
        assert len(venue.submitted) == 1


def test_transient_error_gets_exactly_one_retry(session_factory):
    venue = FakeVenue()
    venue.fail_with, venue.fail_times = BrokerError("503"), 1
    deps, mode = _deps(session_factory, venue)
    symbol = _symbol()
    out = execute({"cycle_id": _cycle(), "approved_orders": [_order(symbol)]}, deps=deps)
    assert out["execution_results"][0]["status"] == "filled"
    assert len(venue.submitted) == 2
    assert len(_fills(session_factory, symbol, mode)) == 1

    venue = FakeVenue()
    venue.fail_with, venue.fail_times = BrokerError("503"), 2
    deps, _ = _deps(session_factory, venue)
    out = execute({"cycle_id": _cycle(), "approved_orders": [_order(_symbol())]}, deps=deps)
    assert out["execution_results"][0]["status"] == "failed"
    assert len(venue.submitted) == 2  # one retry, then give up


def test_unprotected_position_is_not_retried_but_its_entry_is_booked(session_factory):
    venue = FakeVenue()
    deps, mode = _deps(session_factory, venue)
    symbol = _symbol()
    naked_id = f"naked-{uuid.uuid4().hex[:6]}"
    venue.orders[naked_id] = VenueOrder(
        status="filled",
        terminal=True,
        filled_quantity=Decimal("10"),
        fill_price=Decimal("100"),
        filled_at=datetime.now(UTC),
    )
    venue.fail_with = UnprotectedPositionError("stop failed", broker_order_id=naked_id)
    venue.fail_times = 5

    out = execute({"cycle_id": _cycle(), "approved_orders": [_order(symbol)]}, deps=deps)

    assert out["execution_results"][0]["status"] == "unprotected_position"
    assert len(venue.submitted) == 1
    (fill,) = _fills(session_factory, symbol, mode)
    assert fill.quantity == Decimal("10")
    (row,) = _orders(session_factory, symbol)
    assert row.broker_order_id == naked_id and row.status == "filled"


def test_reconcile_failure_blocks_new_positions_but_not_breach_closes(session_factory):
    """A stale book fails closed for NEW positions at that venue only; a
    breach close is risk-reducing, never depends on the book, and goes
    through. Other venues proceed untouched."""
    stocks, crypto = FakeVenue("stock"), FakeVenue("crypto", name="test-crypto")
    deps, _ = _deps(session_factory, stocks, crypto)
    held = _symbol()
    execute({"cycle_id": _cycle(), "approved_orders": [_order(held)]}, deps=deps)  # open stop row
    stocks.lookup_fails = True

    cycle_id = _cycle()
    breach = {"symbol": held, "asset_class": "stock", "quantity": "10"}
    out = execute(
        {
            "cycle_id": cycle_id,
            "stop_breaches": [breach],
            "approved_orders": [_order(_symbol()), _order("BTC-USD", asset_class="crypto")],
        },
        deps=deps,
    )

    assert out["reconcile_failed"] == ["test"]  # watchdog input, even with no submissions
    statuses = [(r["kind"], r["status"]) for r in out["execution_results"]]
    assert statuses == [
        ("stop_breach", "filled"),
        ("order", "reconcile_failed"),
        ("order", "filled"),
    ]
    assert len(stocks.submitted) == 2 and len(crypto.submitted) == 1
    assert _executor_row(session_factory, cycle_id).detail["reconcile_failed"] == ["test"]


def test_lost_venue_order_is_retired_and_never_blocks_the_venue(session_factory):
    """A row the venue has no record of must not poison every later cycle."""
    venue = FakeVenue(sync=False)
    deps, _ = _deps(session_factory, venue)
    symbol = _symbol()
    execute({"cycle_id": _cycle(), "approved_orders": [_order(symbol)]}, deps=deps)
    del venue.orders[venue.oid(1)]  # venue forgot the entry (or we booked a bad id)

    out = execute({"cycle_id": _cycle(), "approved_orders": [_order(_symbol())]}, deps=deps)

    assert out["execution_results"][0]["status"] == "accepted"  # venue NOT marked failed
    entry, stop = _orders(session_factory, symbol)
    assert entry.status == "lost" and stop.status == "held"
    execute({"cycle_id": _cycle()}, deps=deps)
    assert _orders(session_factory, symbol)[0].status == "lost"  # not re-polled/re-lost


def test_fill_without_time_is_booked_now_and_without_price_is_retired(session_factory):
    venue = FakeVenue(sync=False)
    deps, mode = _deps(session_factory, venue)
    timeless, priceless = _symbol(), _symbol()
    execute(
        {"cycle_id": _cycle(), "approved_orders": [_order(timeless), _order(priceless)]},
        deps=deps,
    )
    venue.fill(venue.oid(1))
    venue.orders[venue.oid(1)] = venue.orders[venue.oid(1)].model_copy(update={"filled_at": None})
    venue.fill(venue.oid(2))
    venue.orders[venue.oid(2)] = venue.orders[venue.oid(2)].model_copy(update={"fill_price": None})

    out = execute({"cycle_id": _cycle(), "approved_orders": [_order(_symbol())]}, deps=deps)

    (fill,) = _fills(session_factory, timeless, mode)
    assert fill.filled_at is not None
    assert _fills(session_factory, priceless, mode) == []
    assert _orders(session_factory, priceless)[0].status == "lost"
    assert out["execution_results"][0]["status"] == "accepted"  # venue still usable


def test_drawdown_kill_switch_halts_but_never_cancels_protective_stops(session_factory):
    """D-020: the gate's automatic max-drawdown kill switch stops new entries but
    must leave the venue-side ATR stops working — only the human /killswitch
    cancels everywhere."""
    stocks, crypto = FakeVenue("stock"), FakeVenue("crypto", name="test-crypto")
    redis = FakeRedis()
    redis.set(KILL_SWITCH_KEY, "max_drawdown_exceeded")
    deps, _ = _deps(session_factory, stocks, crypto, redis=redis)
    cycle_id = _cycle()

    out = execute({"cycle_id": cycle_id, "approved_orders": [_order(_symbol())]}, deps=deps)

    assert out["execution_results"][0]["status"] == "halted"
    assert (stocks.cancel_calls, crypto.cancel_calls) == (0, 0)
    assert _executor_row(session_factory, cycle_id).detail["cancelled"] is None


# --- re-protect: when it may run, and what it does without a price or a book ------


class _StopVenue:
    """A venue that accepts standalone stops."""

    name = "venue"

    def __init__(self) -> None:
        """No stops yet."""
        self.placed: list[tuple] = []

    def place_stop(self, symbol, quantity, stop_price, client_order_id):
        """Record the stop and return its id."""
        self.placed.append((symbol, quantity, stop_price))
        return "stop-1"


def _held(symbol: str = "AAPL") -> BrokerPosition:
    """A 5-share position."""
    return BrokerPosition(symbol=symbol, asset_class="stock", quantity=Decimal(5))


def test_a_stop_placed_but_not_booked_is_reported_not_raised(monkeypatch):
    """The stop is real at the venue: a DB failure afterwards must not crash
    the executor (it would skip every submission and the audit row)."""
    venue = _StopVenue()
    candles = [
        Candle(
            symbol="AAPL",
            asset_class="stock",
            ts=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i),
            open=Decimal(100),
            high=Decimal(101),
            low=Decimal(99),
            close=Decimal(100),
            volume=Decimal(1),
            source="itest",
        )
        for i in range(20)
    ]

    class _Repo:
        def __init__(self, session):
            """Bind nothing."""

        def recent_daily_candles(self, symbol, limit):
            """Serve the canned candles."""
            return candles

    class _BrokenBook:
        def __init__(self, session):
            """Bind nothing."""

        def ensure(self, **kwargs):
            """Fail like a DB outage."""
            raise RuntimeError("db down")

    monkeypatch.setattr(executor_mod, "OhlcvRepository", _Repo)
    monkeypatch.setattr(executor_mod, "OrderRepository", _BrokenBook)
    deps = SimpleNamespace(session_factory=contextlib.nullcontext)
    result = executor_mod._protect(
        deps, venue, _held(), Decimal(5), Decimal(100), {"cycle_id": "c"}, datetime.now(UTC), "p"
    )
    assert result["status"] == "placed_unbooked"
    assert venue.placed == [("AAPL", Decimal(5), Decimal(94))]  # 100 - 3 x ATR 2


@pytest.mark.parametrize(("closing", "protected"), [({"AAPL"}, []), (set(), ["AAPL"])])
def test_a_position_this_cycle_closes_gets_no_new_stop(monkeypatch, closing, protected):
    """A sell that goes out this cycle needs no stop; one held back by a halt does."""

    class _EmptyBook:
        def __init__(self, session):
            """Bind nothing."""

        def open_orders(self, broker, mode):
            """No stops booked."""
            return []

        def entry_stop(self, symbol, mode):
            """No app-held stop."""
            return None

    venue = SimpleNamespace(name="v", get_positions=lambda: [_held()])
    calls: list[str] = []
    monkeypatch.setattr(executor_mod, "OrderRepository", _EmptyBook)
    monkeypatch.setattr(executor_mod, "CashFlowRepository", lambda s: SimpleNamespace(ledger=list))
    monkeypatch.setattr(
        executor_mod, "_protect", lambda deps, broker, position, *a: calls.append(position.symbol)
    )
    state = {
        "cycle_id": "c",
        "market_data": {"prices": {"AAPL": Decimal(100)}},
        "stop_breaches": [{"symbol": "AAPL"}],
    }
    deps = SimpleNamespace(brokers=[venue], session_factory=contextlib.nullcontext)
    executor_mod._reprotect(deps, state, closing, set(), datetime.now(UTC), "p")
    assert calls == protected


@pytest.mark.parametrize(
    "reason", ["cooldown:daily_loss", "cooldown:cycle_crashed", "kill_switch:max_drawdown_exceeded"]
)
def test_an_app_held_stop_sale_goes_out_under_halts_that_stop_only_swing_entries(
    session_factory, reason
):
    """Stop-breach sales are protective (D-054): they go out under a breaker cooldown, the
    scheduler's pause and the drawdown switch, while a new swing buy is held back. The
    manual kill switch holds both (test_halted_submits_nothing_but_still_reconciles)."""
    redis = FakeRedis()
    kind, _, value = reason.partition(":")
    redis.set(COOLDOWN_KEY if kind == "cooldown" else KILL_SWITCH_KEY, value, ex=3600)
    venue = FakeVenue()
    deps, _ = _deps(session_factory, venue, redis=redis)
    breach = {"symbol": _symbol(), "asset_class": "stock", "quantity": "0.18"}

    out = execute(
        {"cycle_id": _cycle(), "approved_orders": [_order(_symbol())], "stop_breaches": [breach]},
        deps=deps,
    )

    status = {r["kind"]: r["status"] for r in out["execution_results"]}
    assert status["order"] == "halted" and status["stop_breach"] != "halted"
    assert len(venue.submitted) == 1


@pytest.mark.parametrize(
    ("kind", "symbol", "halted", "protective", "out"),
    [
        ("stop_breach", "AAPL", False, True, True),
        ("stop_breach", "AAPL", True, True, True),  # a cooldown: protective sale
        ("stop_breach", "AAPL", True, False, False),  # the manual kill switch
        ("order", "AAPL", True, True, False),  # a swing entry under a cooldown
        ("order", "VOO", True, True, True),  # the core under a cooldown
    ],
)
def test_goes_out_holds_back_only_what_the_halt_stops(kind, symbol, halted, protective, out):
    """Stop sales and core orders go out under any halt but one that stops everything."""
    assert executor_mod._goes_out(kind, _order(symbol), halted, protective) is out
