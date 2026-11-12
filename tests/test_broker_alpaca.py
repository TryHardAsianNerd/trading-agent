"""AlpacaBroker tests with a mocked SDK client. No network, no real keys.

Venue semantics pinned here: sells as cancel-then-close, no sell rounding or dust,
stops quantize UP, cancel counting confirms < 400 statuses only, and
client_order_id idempotency.
"""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import OrderStatus

from trading_agent.brokers.alpaca import AlpacaBroker
from trading_agent.brokers.base import (
    BrokerError,
    UnfillableQuantityError,
    UnknownOrderError,
    WrongVenueError,
)
from trading_agent.config import LiveTradingNotConfirmedError, Settings
from trading_agent.graph.state import ApprovedOrder, TradeProposal


def _settings(**kw: str) -> Settings:
    defaults = {"alpaca_api_key": "key", "alpaca_secret_key": "secret"}
    defaults.update(kw)
    return Settings(_env_file=None, **defaults)


def _order(
    side: str = "buy",
    quantity: str = "50",
    stop: str | None = "96.00",
    asset_class: str = "stock",
    symbol: str = "AAPL",
) -> ApprovedOrder:
    return ApprovedOrder(
        symbol=symbol,
        asset_class=asset_class,
        side=side,
        quantity=Decimal(quantity),
        stop_loss_price=Decimal(stop) if stop is not None else None,
        source_proposal=TradeProposal(
            symbol=symbol, asset_class=asset_class, action=side, confidence=0.8, reasoning="r"
        ),
    )


def _sdk(order_id: str = "abc-123") -> MagicMock:
    client = MagicMock()
    # Real enum for status: production runs the getattr(status, "value") branch.
    client.submit_order.return_value = SimpleNamespace(
        id=order_id,
        status=OrderStatus.ACCEPTED,
        legs=[SimpleNamespace(id=f"{order_id}-leg", status=OrderStatus.HELD)],
    )
    client.get_orders.return_value = []
    client.close_position.return_value = SimpleNamespace(
        id="close-1", status=OrderStatus.ACCEPTED, qty="12.5"
    )
    # Default: no order for any key, every position held — failures stay failures (no replay).
    client.get_order_by_client_id.side_effect = RuntimeError("404 order not found")
    client.get_open_position.return_value = SimpleNamespace(qty="50")
    return client


# --- construction / live-mode friction ---------------------------------------


def test_paper_settings_construct_paper_client(monkeypatch):
    created = {}

    def fake_client(api_key, secret_key, paper):
        created.update(api_key=api_key, secret_key=secret_key, paper=paper)
        return MagicMock()

    monkeypatch.setattr("trading_agent.brokers.alpaca.TradingClient", fake_client)
    AlpacaBroker(_settings())
    assert created["paper"] is True


def test_live_without_confirmation_refused():
    fake_settings = SimpleNamespace(
        trading_mode="live",
        live_trading_confirmed="",
        alpaca_api_key="k",
        alpaca_secret_key="s",
        is_paper=False,
    )
    with pytest.raises(LiveTradingNotConfirmedError):
        AlpacaBroker(fake_settings, client=MagicMock())


def test_guard_uses_endpoint_predicate_not_mode_string():
    """A duck-typed settings object with is_paper=False must be refused even if
    trading_mode lies — the guard gates on the predicate that picks the URL."""
    liar = SimpleNamespace(
        trading_mode="paper",  # claims paper...
        live_trading_confirmed="",
        alpaca_api_key="k",
        alpaca_secret_key="s",
        is_paper=False,  # ...but selects the live endpoint
    )
    with pytest.raises(LiveTradingNotConfirmedError):
        AlpacaBroker(liar, client=MagicMock())


def test_missing_keys_refused():
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        AlpacaBroker(_settings(alpaca_api_key=""))


# --- venue routing ------------------------------------------------------------


def test_crypto_order_rejected():
    broker = AlpacaBroker(_settings(), client=_sdk())
    with pytest.raises(WrongVenueError):
        broker.submit_order(_order(asset_class="crypto", symbol="BTC-USD"))


# --- buys -----------------------------------------------------------------------


def test_a_swing_buy_is_a_fractional_day_market_order_with_no_stop_attached():
    """Alpaca takes fractions only on simple DAY orders; the app holds the stop (D-054)."""
    sdk = _sdk()
    broker = AlpacaBroker(_settings(), client=sdk)
    result = broker.submit_order(_order(quantity="0.1812345678"))

    request = sdk.submit_order.call_args.args[0]
    assert request.symbol == "AAPL"
    assert request.qty == pytest.approx(0.181234567)
    assert str(request.side.value) == "buy"
    assert str(request.time_in_force.value) == "day"
    assert request.order_class is None and request.stop_loss is None
    assert result.broker_order_id == "abc-123"
    assert result.status == "accepted"
    assert result.submitted_quantity == Decimal("0.181234567")
    assert result.protective_order_id is None


def test_buy_sends_idempotency_key():
    """Every buy carries a client_order_id so a retry after an
    ambiguous timeout cannot double-submit."""
    sdk = _sdk()
    broker = AlpacaBroker(_settings(), client=sdk)
    result = broker.submit_order(_order(), client_order_id="cycle1-AAPL-buy")
    assert sdk.submit_order.call_args.args[0].client_order_id == "cycle1-AAPL-buy"
    assert result.client_order_id == "cycle1-AAPL-buy"


def test_buy_autogenerates_idempotency_key_when_omitted():
    sdk = _sdk()
    result = AlpacaBroker(_settings(), client=sdk).submit_order(_order())
    sent = sdk.submit_order.call_args.args[0].client_order_id
    assert sent and sent == result.client_order_id


def test_a_stopless_core_buy_is_fractional_and_day():
    """D-041: the core buys fractional units with no venue stop; Alpaca takes
    fractional quantities on DAY orders only."""
    sdk = _sdk()
    result = AlpacaBroker(_settings(), client=sdk).submit_order(
        _order(quantity="0.1234567891234", stop=None, symbol="VOO")
    )
    request = sdk.submit_order.call_args.args[0]
    assert request.qty == pytest.approx(0.123456789)
    assert str(request.time_in_force.value) == "day"
    assert request.order_class is None and request.stop_loss is None
    assert result.submitted_quantity == Decimal("0.123456789")


def test_a_buy_under_alpacas_precision_is_unfillable():
    with pytest.raises(UnfillableQuantityError):
        AlpacaBroker(_settings(), client=_sdk()).submit_order(_order(quantity="0.0000000004"))


def test_subdollar_stop_uses_fine_tick():
    """Sub-$1 stops use Alpaca's $0.0001 increment and cannot quantize to zero."""
    sdk = _sdk()
    AlpacaBroker(_settings(), client=sdk).place_stop("AAPL", Decimal(5), Decimal("0.00412"), "k")
    assert sdk.submit_order.call_args.args[0].stop_price == 0.0042


def test_sdk_failure_wrapped_as_broker_error():
    sdk = _sdk()
    sdk.submit_order.side_effect = RuntimeError("api down")
    with pytest.raises(BrokerError, match="submit_order failed"):
        AlpacaBroker(_settings(), client=sdk).submit_order(_order())


# --- sells: cancel-then-close ----------------------------------------------------


def test_sell_cancels_symbol_orders_then_closes_position():
    """A working stop holds the shares, so a plain market sell would be rejected:
    sells cancel the symbol's open orders first, then close venue-side."""
    sdk = _sdk()
    sdk.get_orders.return_value = [SimpleNamespace(id="stop-leg-1")]
    broker = AlpacaBroker(_settings(), client=sdk)
    result = broker.submit_order(_order(side="sell", stop=None, quantity="12.5"))

    filt = sdk.get_orders.call_args.args[0]
    assert filt.symbols == ["AAPL"]
    assert str(filt.status.value) == "open"
    sdk.cancel_order_by_id.assert_called_once_with("stop-leg-1")
    sdk.close_position.assert_called_once_with("AAPL")
    assert result.broker_order_id == "close-1"


def test_sell_closes_full_fractional_position_no_dust():
    """Close-position handles fractional shares venue-side: 12.5 closes as 12.5,
    never rounded to 12 with 0.5 dust left exposed."""
    sdk = _sdk()
    result = AlpacaBroker(_settings(), client=sdk).submit_order(
        _order(side="sell", stop=None, quantity="12.5")
    )
    assert result.submitted_quantity == Decimal("12.5")


def test_sell_with_stop_rejected_as_malformed():
    """A sell-side OTO stop child is nonsense; reject loudly."""
    with pytest.raises(BrokerError, match="must not carry a stop"):
        AlpacaBroker(_settings(), client=_sdk()).submit_order(_order(side="sell", stop="96.00"))


def test_sell_fails_loud_when_cancel_fails():
    """A sell must not proceed while shares are still reserved by an open order."""
    sdk = _sdk()
    sdk.get_orders.return_value = [SimpleNamespace(id="stop-leg-1")]
    sdk.cancel_order_by_id.side_effect = RuntimeError("500")
    with pytest.raises(BrokerError, match="cancel of AAPL open orders failed"):
        AlpacaBroker(_settings(), client=sdk).submit_order(
            _order(side="sell", stop=None, quantity="5")
        )
    sdk.close_position.assert_not_called()


# --- cancel-all -------------------------------------------------------------------


def test_cancel_all_counts_only_confirmed_cancellations():
    """The SDK returns per-order HTTP statuses for ATTEMPTS; a 500 means a live
    order remains and must not be counted as cancelled."""
    sdk = _sdk()
    sdk.cancel_orders.return_value = [
        SimpleNamespace(id="a", status=200),
        SimpleNamespace(id="b", status=500),
        SimpleNamespace(id="c", status=204),
    ]
    assert AlpacaBroker(_settings(), client=sdk).cancel_all_orders() == 2


# --- account state -----------------------------------------------------------------


def test_positions_normalized_to_decimal():
    sdk = _sdk()
    sdk.get_all_positions.return_value = [
        SimpleNamespace(symbol="AAPL", qty="12.5", avg_entry_price="187.33")
    ]
    (pos,) = AlpacaBroker(_settings(), client=sdk).get_positions()
    assert pos.quantity == Decimal("12.5")
    assert pos.avg_entry_price == Decimal("187.33")
    assert pos.asset_class == "stock"


def test_account_summary_decimals():
    sdk = _sdk()
    sdk.get_account.return_value = SimpleNamespace(equity="100000.55", cash="40000.25")
    summary = AlpacaBroker(_settings(), client=sdk).get_account()
    assert summary.equity == Decimal("100000.55")
    assert summary.cash == Decimal("40000.25")


def test_account_missing_fields_is_error():
    sdk = _sdk()
    sdk.get_account.return_value = SimpleNamespace(equity=None, cash=None)
    with pytest.raises(BrokerError, match="missing equity/cash"):
        AlpacaBroker(_settings(), client=sdk).get_account()


def test_declares_venue_name_and_asset_class():
    broker = AlpacaBroker(_settings(), client=_sdk())
    assert (broker.name, broker.asset_class) == ("alpaca", "stock")


def test_failed_buy_is_replayed_when_the_venue_already_holds_the_key():
    """Alpaca REJECTS a reused client_order_id (422) instead of returning
    the original; a lost response or a resumed cycle must still yield the
    FIRST order, never a failure that orphans it."""
    sdk = _sdk()
    sdk.submit_order.side_effect = RuntimeError("422 client_order_id must be unique")
    sdk.get_order_by_client_id.side_effect = None
    sdk.get_order_by_client_id.return_value = SimpleNamespace(
        id="orig-1", status=OrderStatus.FILLED, legs=[SimpleNamespace(id="orig-1-leg")]
    )

    submitted = AlpacaBroker(_settings(), client=sdk).submit_order(_order(), "cycle:AAPL")

    assert submitted.broker_order_id == "orig-1"
    assert submitted.status == "filled"
    sdk.get_order_by_client_id.assert_called_once_with("cycle:AAPL")


def test_failed_buy_with_no_venue_order_for_the_key_is_a_broker_error():
    sdk = _sdk()
    sdk.submit_order.side_effect = RuntimeError("503")
    with pytest.raises(BrokerError, match="submit_order failed"):
        AlpacaBroker(_settings(), client=sdk).submit_order(_order(), "cycle:AAPL")


def test_failed_close_is_replayed_from_latest_filled_sell_once_position_is_gone():
    sdk = _sdk()
    sdk.close_position.side_effect = RuntimeError("404 position does not exist")
    sdk.get_open_position.side_effect = RuntimeError("404 position does not exist")
    sdk.get_orders.side_effect = [
        [],  # _cancel_symbol_orders: nothing open
        [
            SimpleNamespace(id="old-cancel", status=OrderStatus.CANCELED, filled_qty="0", qty="0"),
            SimpleNamespace(id="sell-9", status=OrderStatus.FILLED, filled_qty="12.5", qty="12.5"),
        ],
    ]

    submitted = AlpacaBroker(_settings(), client=sdk).submit_order(_order(side="sell", stop=None))

    assert submitted.broker_order_id == "sell-9"
    assert submitted.submitted_quantity == Decimal("12.5")


def test_failed_close_with_position_still_held_is_a_broker_error():
    sdk = _sdk()
    sdk.close_position.side_effect = RuntimeError("503")
    with pytest.raises(BrokerError, match="close_position failed"):
        AlpacaBroker(_settings(), client=sdk).submit_order(_order(side="sell", stop=None))
    sdk.get_orders.assert_called_once()  # only the cancel listing, no replay search


def test_get_order_404_is_an_unknown_order_error():
    sdk = _sdk()

    class NotFound(Exception):
        status_code = 404

    sdk.get_order_by_id.side_effect = NotFound("order not found")
    with pytest.raises(UnknownOrderError):
        AlpacaBroker(_settings(), client=sdk).get_order("o-gone")


def _venue(status, filled_qty="0", price=None, filled_at=None):
    return SimpleNamespace(
        id="o-1", status=status, filled_qty=filled_qty, filled_avg_price=price, filled_at=filled_at
    )


def test_get_order_maps_filled_to_terminal_with_price_and_time():
    from datetime import UTC, datetime

    sdk = _sdk()
    when = datetime(2026, 9, 15, 14, 30, tzinfo=UTC)
    sdk.get_order_by_id.return_value = _venue(OrderStatus.FILLED, "50", "101.25", when)
    venue = AlpacaBroker(_settings(), client=sdk).get_order("o-1")
    assert venue.terminal and venue.status == "filled"
    assert venue.filled_quantity == Decimal("50")
    assert venue.fill_price == Decimal("101.25")
    assert venue.filled_at == when and venue.fees == 0


def test_get_order_maps_open_and_cancelled_states():
    sdk = _sdk()
    sdk.get_order_by_id.return_value = _venue(OrderStatus.HELD)
    held = AlpacaBroker(_settings(), client=sdk).get_order("o-1")
    assert not held.terminal and held.filled_quantity == 0 and held.fill_price is None

    sdk.get_order_by_id.return_value = _venue(OrderStatus.CANCELED)
    cancelled = AlpacaBroker(_settings(), client=sdk).get_order("o-1")
    assert cancelled.terminal and cancelled.filled_quantity == 0

    # done_for_day is NOT terminal for GTC orders: they resume next session.
    sdk.get_order_by_id.return_value = _venue(OrderStatus.DONE_FOR_DAY)
    assert not AlpacaBroker(_settings(), client=sdk).get_order("o-1").terminal


def test_get_order_wraps_sdk_failures():
    sdk = _sdk()
    sdk.get_order_by_id.side_effect = RuntimeError("504")
    with pytest.raises(BrokerError, match="get_order_by_id failed"):
        AlpacaBroker(_settings(), client=sdk).get_order("o-1")


# --- standalone protective stop ---------------------------------------------------


def test_place_stop_is_a_gtc_whole_share_sell_stop_rounded_up():
    sdk = _sdk("stop-9")
    broker = AlpacaBroker(_settings(), client=sdk)
    assert broker.place_stop("AAPL", Decimal("7.6"), Decimal("94.123"), "c1:AAPL:protect") == (
        "stop-9"
    )
    request = sdk.submit_order.call_args.args[0]
    assert (request.symbol, request.qty, str(request.side.value)) == ("AAPL", 7, "sell")
    assert str(request.time_in_force.value) == "gtc"
    assert request.stop_price == 94.13  # tighter, never looser
    assert request.client_order_id == "c1:AAPL:protect"


def test_place_stop_under_one_share_is_unfillable():
    with pytest.raises(UnfillableQuantityError):
        AlpacaBroker(_settings(), client=_sdk()).place_stop(
            "AAPL", Decimal("0.5"), Decimal(90), "k"
        )


def test_place_stop_replays_a_rejected_duplicate_key():
    sdk = _sdk()
    sdk.submit_order.side_effect = RuntimeError("422 client_order_id must be unique")
    sdk.get_order_by_client_id.side_effect = None
    sdk.get_order_by_client_id.return_value = SimpleNamespace(id="orig-1")
    broker = AlpacaBroker(_settings(), client=sdk)
    assert broker.place_stop("AAPL", Decimal(5), Decimal(90), "k") == "orig-1"


def test_place_stop_failure_without_replay_raises_broker_error():
    sdk = _sdk()
    sdk.submit_order.side_effect = RuntimeError("503")
    with pytest.raises(BrokerError):
        AlpacaBroker(_settings(), client=sdk).place_stop("AAPL", Decimal(5), Decimal(90), "k")
