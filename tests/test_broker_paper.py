"""PaperBroker tests against the real compose TimescaleDB (D-013).

Skipped when the database is unreachable. Venue state must live in the DB
(restart survival is a project invariant), so these tests run the real
persistence path; determinism comes from the fixed quote fake and the
constant slippage model. Accounts use unique "t-*" keys and are cleaned up.
"""

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import text

from trading_agent.brokers.base import (
    BrokerError,
    UnfillableQuantityError,
    UnknownOrderError,
    WrongVenueError,
)
from trading_agent.brokers.paper import (
    FILL_COST_PCT,
    SLIPPAGE_PCT,
    PaperBroker,
    PaperOnlyBrokerError,
)
from trading_agent.config import Settings
from trading_agent.graph.state import ApprovedOrder, TradeProposal
from trading_agent.storage.db import init_db, make_engine, make_session_factory
from trading_agent.storage.models import PaperOrder


def _engine_or_skip():
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
        for table in ("paper_orders", "paper_positions", "paper_balances"):
            conn.execute(text(f"DELETE FROM {table} WHERE account LIKE 't-%'"))


@pytest.fixture()
def session_factory(engine):
    return make_session_factory(engine)


class Quotes:
    """Mutable quote fake: tests move the market by assigning prices."""

    def __init__(self, **prices):
        self.prices = {k.replace("_", "-"): Decimal(v) for k, v in prices.items()}
        self.fail = False

    def __call__(self, symbol):
        if self.fail:
            raise RuntimeError("quote source down")
        return self.prices[symbol]


def _order(symbol="BTC-USD", side="buy", quantity="1", stop="96", asset_class="crypto"):
    return ApprovedOrder(
        symbol=symbol,
        asset_class=asset_class,
        side=side,
        quantity=Decimal(quantity),
        stop_loss_price=Decimal(stop) if stop is not None else None,
        source_proposal=TradeProposal(
            symbol=symbol,
            asset_class="crypto" if asset_class == "crypto" else "stock",
            action=side,
            confidence=0.5,
            reasoning="test",
        ),
    )


def _broker(session_factory, quotes, cash="10000"):
    settings = Settings(_env_file=None, paper_starting_cash=cash)
    account = f"t-{uuid.uuid4().hex[:8]}"
    return PaperBroker(settings, session_factory, quotes, account=account), account


def test_slippage_constant_is_within_risk_rules_band():
    assert Decimal("0.001") <= SLIPPAGE_PCT <= Decimal("0.005")


def test_refuses_construction_outside_paper_mode(session_factory):
    live = Settings.model_construct(trading_mode="live")
    with pytest.raises(PaperOnlyBrokerError):
        PaperBroker(live, session_factory, Quotes(BTC_USD="100"))


def test_buy_fills_with_slippage_and_records_stop(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)

    submitted = broker.submit_order(_order(quantity="10"), client_order_id="c1")

    assert submitted.status == "filled"
    assert submitted.filled_quantity == Decimal("10")
    fill = Decimal("100") * (1 + FILL_COST_PCT)
    assert fill == Decimal("100.8")  # 0.2% slippage + 0.6% Coinbase taker fee (D-030)
    account = broker.get_account()
    assert account.cash == Decimal("10000") - fill * 10
    (position,) = broker.get_positions()
    assert position.symbol == "BTC-USD"
    assert position.quantity == Decimal("10")
    assert position.avg_entry_price == fill  # lots reported: tracker uses them
    # equity = cash + qty * current quote
    assert account.equity == account.cash + Decimal("10") * Decimal("100")


def test_client_order_id_retry_never_double_fills(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)

    first = broker.submit_order(_order(quantity="5"), client_order_id="retry-me")
    second = broker.submit_order(_order(quantity="5"), client_order_id="retry-me")

    assert second.broker_order_id == first.broker_order_id
    (position,) = broker.get_positions()
    assert position.quantity == Decimal("5")  # once, not twice


def test_second_buy_averages_cost(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10"))
    quotes.prices["BTC-USD"] = Decimal("200")
    broker.submit_order(_order(quantity="10", stop="190"))

    (position,) = broker.get_positions()
    assert position.quantity == Decimal("20")
    expected_avg = (Decimal("100") + Decimal("200")) / 2 * (1 + FILL_COST_PCT)
    assert position.avg_entry_price == expected_avg


def test_sell_is_full_close_and_cancels_stops(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, account = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10"))

    broker.submit_order(_order(side="sell", quantity="10", stop=None))

    assert broker.get_positions() == []
    proceeds = Decimal("100") * (1 - FILL_COST_PCT) * 10
    cost = Decimal("100") * (1 + FILL_COST_PCT) * 10
    assert broker.get_account().cash == Decimal("10000") - cost + proceeds
    with session_factory() as session:
        stops = (
            session.query(PaperOrder)
            .filter(PaperOrder.account == account, PaperOrder.order_type == "stop")
            .all()
        )
    assert [s.status for s in stops] == ["cancelled"]


def test_stop_triggers_on_observation_at_observed_quote(session_factory):
    """The market gaps THROUGH the stop (96) to 90: the stop fills at the
    observed quote minus slippage — gaps fill worse, matching reality."""
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10", stop="96"))

    quotes.prices["BTC-USD"] = Decimal("90")
    positions = broker.get_positions()  # observation settles the stop

    assert positions == []
    proceeds = Decimal("90") * (1 - FILL_COST_PCT) * 10
    cost = Decimal("100") * (1 + FILL_COST_PCT) * 10
    assert broker.get_account().cash == Decimal("10000") - cost + proceeds


def test_stop_does_not_trigger_above_stop_price(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10", stop="96"))

    quotes.prices["BTC-USD"] = Decimal("97")
    (position,) = broker.get_positions()
    assert position.quantity == Decimal("10")


def test_quote_failure_during_settle_keeps_stop_open(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10", stop="96"))

    quotes.fail = True
    (position,) = broker.get_positions()  # settle skips, never raises
    assert position.quantity == Decimal("10")

    quotes.fail = False
    quotes.prices["BTC-USD"] = Decimal("95")
    assert broker.get_positions() == []  # stop still armed, fires next look


def test_stacked_stops_fire_partially_and_leave_residual_armed(session_factory):
    """Two buys stack two stops at different triggers. A drop through only
    the higher one closes that lot's quantity; the residual position keeps
    its own stop armed. Cash must be exact through the partial fire."""
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10", stop="96"))
    quotes.prices["BTC-USD"] = Decimal("110")
    broker.submit_order(_order(quantity="5", stop="104"))
    cash_after_buys = broker.get_account().cash

    quotes.prices["BTC-USD"] = Decimal("103")  # crosses 104 stop, not 96
    (position,) = broker.get_positions()

    assert position.quantity == Decimal("10")  # 15 - the 5-lot stop
    fired_proceeds = Decimal("103") * (1 - FILL_COST_PCT) * 5
    assert broker.get_account().cash == cash_after_buys + fired_proceeds

    quotes.prices["BTC-USD"] = Decimal("95")  # residual stop still armed
    assert broker.get_positions() == []


def test_get_account_also_settles_stops(session_factory):
    """Both observation points settle — the tracker calls get_account first,
    so a crossed stop must not survive an account read."""
    quotes = Quotes(BTC_USD="100")
    broker, account = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10", stop="96"))

    quotes.prices["BTC-USD"] = Decimal("90")
    summary = broker.get_account()

    proceeds = Decimal("90") * (1 - FILL_COST_PCT) * 10
    cost = Decimal("100") * (1 + FILL_COST_PCT) * 10
    assert summary.cash == Decimal("10000") - cost + proceeds
    assert summary.equity == summary.cash  # flat after the stop fill
    with session_factory() as session:
        (stop,) = (
            session.query(PaperOrder)
            .filter(PaperOrder.account == account, PaperOrder.order_type == "stop")
            .all()
        )
    assert stop.status == "filled"
    assert stop.quantity == Decimal("10")  # original size kept (audit trail)
    assert stop.fill_price == Decimal("90") * (1 - FILL_COST_PCT)


def test_restart_survives_process_death(session_factory):
    """Project invariant: venue state in the DB. A brand-new broker instance
    over the same account sees identical cash, positions, and open stops."""
    quotes = Quotes(BTC_USD="100")
    settings = Settings(_env_file=None)
    account = f"t-{uuid.uuid4().hex[:8]}"
    first = PaperBroker(settings, session_factory, quotes, account=account)
    first.submit_order(_order(quantity="10", stop="96"))
    cash_before = first.get_account().cash

    reborn = PaperBroker(settings, session_factory, quotes, account=account)
    assert reborn.get_account().cash == cash_before
    (position,) = reborn.get_positions()
    assert position.quantity == Decimal("10")
    quotes.prices["BTC-USD"] = Decimal("95")
    assert reborn.get_positions() == []  # the stop survived the "restart" too


def test_permanent_and_transient_errors(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes, cash="100")

    with pytest.raises(WrongVenueError):
        broker.submit_order(_order(symbol="AAPL", asset_class="stock"))
    with pytest.raises(UnfillableQuantityError):
        broker.submit_order(_order(quantity="0"))
    with pytest.raises(BrokerError, match="insufficient simulated cash"):
        broker.submit_order(_order(quantity="10"))  # needs 1002, has 100
    with pytest.raises(BrokerError, match="no simulated position"):
        broker.submit_order(_order(side="sell", quantity="1", stop=None))
    quotes.fail = True
    with pytest.raises(BrokerError, match="paper quote failed"):
        broker.submit_order(_order(quantity="0.01"))


def test_mark_falls_back_to_entry_when_quote_fails(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10", stop=None))

    quotes.fail = True
    account = broker.get_account()
    fill = Decimal("100") * (1 + FILL_COST_PCT)
    assert account.equity == account.cash + Decimal("10") * fill  # marked at entry


def test_cancel_all_orders_counts_open_stops(session_factory):
    quotes = Quotes(BTC_USD="100", ETH_USD="50")
    broker, _ = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="1", stop="96"))
    broker.submit_order(_order(symbol="ETH-USD", quantity="1", stop="45"))

    assert broker.cancel_all_orders() == 2
    assert broker.cancel_all_orders() == 0  # already cancelled
    quotes.prices["BTC-USD"] = Decimal("1")
    positions = sorted(broker.get_positions(), key=lambda p: p.symbol)
    assert len(positions) == 2  # cancelled stops cannot fire
    assert positions[0].quantity == Decimal("1")


def test_declares_venue_name_and_asset_class(session_factory):
    broker, _ = _broker(session_factory, Quotes(BTC_USD="100"))
    assert (broker.name, broker.asset_class) == ("paper", "crypto")


def test_buy_returns_protective_id_and_get_order_reports_fills(session_factory):
    """The executor books the entry AND its stop by venue id; get_order is
    the reconciliation seam for both (fill fields on the entry, open on
    the stop until it fires)."""
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)

    submitted = broker.submit_order(_order(quantity="10", stop="96"), client_order_id="c1")

    assert submitted.protective_order_id is not None
    entry = broker.get_order(submitted.broker_order_id)
    assert entry.terminal and entry.status == "filled"
    assert entry.filled_quantity == Decimal("10")
    assert entry.fill_price == Decimal("100") * (1 + FILL_COST_PCT)
    assert entry.filled_at is not None
    stop = broker.get_order(submitted.protective_order_id)
    assert not stop.terminal and stop.filled_quantity == 0 and stop.fill_price is None

    quotes.prices["BTC-USD"] = Decimal("90")
    broker.get_positions()  # observation settles the stop
    fired = broker.get_order(submitted.protective_order_id)
    assert fired.terminal and fired.status == "filled"
    assert fired.filled_quantity == Decimal("10")
    assert fired.fill_price == Decimal("90") * (1 - FILL_COST_PCT)
    assert fired.filled_at is not None


def test_replay_returns_the_same_protective_id(session_factory):
    """A resumed cycle re-submits with the same key: the book must see the
    SAME stop id, not None (an unbooked stop's fill would never reconcile)."""
    broker, _ = _broker(session_factory, Quotes(BTC_USD="100"))
    first = broker.submit_order(_order(quantity="5"), client_order_id="again")
    second = broker.submit_order(_order(quantity="5"), client_order_id="again")
    assert second.protective_order_id == first.protective_order_id is not None


def test_stacked_stop_partial_fire_reports_actual_filled_quantity(session_factory):
    """The stop row keeps its requested size for audit; get_order reports
    what actually filled (capped at the remaining position)."""
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)
    first = broker.submit_order(_order(quantity="10", stop="96"), client_order_id="a")
    quotes.prices["BTC-USD"] = Decimal("110")
    second = broker.submit_order(_order(quantity="5", stop="104"), client_order_id="b")

    quotes.prices["BTC-USD"] = Decimal("95")  # crosses both; 15 held in total
    broker.get_positions()

    fills = [broker.get_order(s.protective_order_id).filled_quantity for s in (first, second)]
    assert sorted(fills) == [Decimal("5"), Decimal("10")]


def test_get_order_rejects_unknown_or_foreign_ids(session_factory):
    broker, _ = _broker(session_factory, Quotes(BTC_USD="100"))
    other, _ = _broker(session_factory, Quotes(BTC_USD="100"))
    submitted = other.submit_order(_order(quantity="1"), client_order_id="x")
    with pytest.raises(UnknownOrderError):
        broker.get_order(submitted.broker_order_id)  # another account's row
    with pytest.raises(UnknownOrderError):
        broker.get_order("not-a-number")
    with pytest.raises(UnknownOrderError):
        broker.get_order("999999999")


# --- standalone protective stop ---------------------------------------------------


def test_place_stop_protects_a_held_position_and_fires_like_any_stop(session_factory):
    quotes = Quotes(BTC_USD="100")
    broker, _ = _broker(session_factory, quotes)
    broker.submit_order(_order(quantity="10", stop=None), client_order_id="c1")  # naked
    stop_id = broker.place_stop("BTC-USD", Decimal(10), Decimal(94), "c2:BTC-USD:protect")
    # Idempotent by key: a resumed cycle gets the same stop back.
    assert broker.place_stop("BTC-USD", Decimal(10), Decimal(94), "c2:BTC-USD:protect") == stop_id
    assert broker.get_order(stop_id).terminal is False

    quotes.prices["BTC-USD"] = Decimal(93)
    assert broker.get_positions() == []  # settled at the observed quote
    fired = broker.get_order(stop_id)
    assert fired.terminal and fired.filled_quantity == Decimal(10)
    assert fired.fill_price == Decimal(93) * (1 - FILL_COST_PCT)


def test_place_stop_refuses_a_nonpositive_quantity(session_factory):
    broker, _ = _broker(session_factory, Quotes(BTC_USD="100"))
    with pytest.raises(UnfillableQuantityError):
        broker.place_stop("BTC-USD", Decimal(0), Decimal(90), "k")
