"""CoinbaseBroker tests with a mocked SDK client. No network, no real keys.

Pins the venue semantics: terminal-status polling before the stop is sized, holds
counted in positions and equity, cancel-then-await-release-then-sell,
UnprotectedPositionError typing, sell idempotency keys, pagination, base_min_size.
"""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trading_agent.brokers.base import (
    BrokerError,
    UnfillableQuantityError,
    UnknownOrderError,
    UnprotectedPositionError,
    WrongVenueError,
)
from trading_agent.brokers.coinbase import CoinbaseBroker
from trading_agent.config import LiveTradingNotConfirmedError, Settings
from trading_agent.graph.state import ApprovedOrder, TradeProposal


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Poll loops must not slow the suite down."""
    monkeypatch.setattr("trading_agent.brokers.coinbase.time.sleep", lambda _s: None)


def _settings(**kw: str) -> Settings:
    defaults = {"coinbase_api_key": "key", "coinbase_api_secret": "secret"}
    defaults.update(kw)
    return Settings(_env_file=None, **defaults)


def _order(
    side: str = "buy",
    quantity: str = "0.5",
    stop: str | None = "94000.00",
    asset_class: str = "crypto",
    symbol: str = "BTC-USD",
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


def _ok_create(order_id: str = "cb-1") -> SimpleNamespace:
    return SimpleNamespace(success=True, success_response={"order_id": order_id})


def _venue_order(status: str = "FILLED", filled: str = "0.5") -> SimpleNamespace:
    return SimpleNamespace(order=SimpleNamespace(status=status, filled_size=filled))


def _sdk() -> MagicMock:
    client = MagicMock()
    client.get_product.return_value = SimpleNamespace(
        price="97000.00",
        base_increment="0.00000001",
        quote_increment="0.01",
        base_min_size="0.00001",
    )
    client.market_order_buy.return_value = _ok_create("entry-1")
    client.stop_limit_order_gtc_sell.return_value = _ok_create("stop-1")
    client.market_order_sell.return_value = _ok_create("sell-1")
    client.get_order.return_value = _venue_order()
    client.list_orders.return_value = SimpleNamespace(orders=[], has_next=False)
    client.cancel_orders.return_value = SimpleNamespace(results=[])
    return client


# --- construction / live-mode friction ---------------------------------------


def test_paper_settings_use_sandbox_url(monkeypatch):
    created = {}

    def fake_client(api_key, api_secret, **kwargs):
        created.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("trading_agent.brokers.coinbase.RESTClient", fake_client)
    CoinbaseBroker(_settings())
    assert created["base_url"] == "api-sandbox.coinbase.com"


def test_live_without_confirmation_refused():
    liar = SimpleNamespace(
        trading_mode="paper",
        live_trading_confirmed="",
        coinbase_api_key="k",
        coinbase_api_secret="s",
        is_paper=False,
    )
    with pytest.raises(LiveTradingNotConfirmedError):
        CoinbaseBroker(liar, client=MagicMock())


def test_missing_keys_refused():
    with pytest.raises(ValueError, match="COINBASE_API_KEY"):
        CoinbaseBroker(_settings(coinbase_api_key=""))


def test_stock_order_rejected():
    with pytest.raises(WrongVenueError):
        CoinbaseBroker(_settings(), client=_sdk()).submit_order(
            _order(asset_class="stock", symbol="AAPL")
        )


# --- buys: terminal-status polling then GTC stop-limit ------------------------


def test_buy_places_market_then_gtc_stop_limit_for_filled_size():
    sdk = _sdk()
    result = CoinbaseBroker(_settings(), client=sdk).submit_order(
        _order(), client_order_id="cycle1-BTC-buy"
    )

    buy_kwargs = sdk.market_order_buy.call_args.kwargs
    assert buy_kwargs["product_id"] == "BTC-USD"
    assert buy_kwargs["base_size"] == "0.50000000"
    assert buy_kwargs["client_order_id"] == "cycle1-BTC-buy"

    stop_kwargs = sdk.stop_limit_order_gtc_sell.call_args.kwargs
    assert stop_kwargs["base_size"] == "0.5"
    assert stop_kwargs["stop_price"] == "94000.00"
    assert stop_kwargs["stop_direction"] == "STOP_DIRECTION_STOP_DOWN"
    assert stop_kwargs["client_order_id"] == "cycle1-BTC-buy-stop"

    assert result.status == "FILLED"  # venue status, passed through
    assert result.filled_quantity == Decimal("0.5")
    assert result.client_order_id == "cycle1-BTC-buy"


def test_buy_polls_until_terminal_before_sizing_stop():
    """An instant filled_size=0 read must NOT skip the stop —
    the fill was still materializing."""
    sdk = _sdk()
    sdk.get_order.side_effect = [
        _venue_order(status="PENDING", filled="0"),
        _venue_order(status="OPEN", filled="0.2"),
        _venue_order(status="FILLED", filled="0.5"),
    ]
    CoinbaseBroker(_settings(), client=sdk).submit_order(_order())
    # Stop sized to the TERMINAL fill, not the instant read.
    assert sdk.stop_limit_order_gtc_sell.call_args.kwargs["base_size"] == "0.5"
    assert sdk.get_order.call_count == 3


def test_buy_tolerates_order_not_yet_queryable():
    """A 404 right after create means 'not visible yet', not 'failed'."""
    sdk = _sdk()
    sdk.get_order.side_effect = [RuntimeError("404"), _venue_order()]
    result = CoinbaseBroker(_settings(), client=sdk).submit_order(_order())
    assert result.status == "FILLED"


def test_buy_never_terminal_is_an_error_not_a_silent_skip():
    """State UNKNOWN must be a loud failure — never treated as 'no fill'."""
    sdk = _sdk()
    sdk.get_order.return_value = _venue_order(status="PENDING", filled="0")
    with pytest.raises(BrokerError, match="state UNKNOWN"):
        CoinbaseBroker(_settings(), client=sdk).submit_order(_order())
    sdk.stop_limit_order_gtc_sell.assert_not_called()


def test_terminally_unfilled_buy_places_no_stop():
    sdk = _sdk()
    sdk.get_order.return_value = _venue_order(status="CANCELLED", filled="0")
    result = CoinbaseBroker(_settings(), client=sdk).submit_order(_order())
    sdk.stop_limit_order_gtc_sell.assert_not_called()
    assert result.status == "CANCELLED"
    assert result.filled_quantity == Decimal(0)


def test_filled_buy_with_failed_stop_raises_typed_unprotected_error():
    """The one non-retryable alert-worthy failure gets its own type — a blind
    entry retry would double the position."""
    sdk = _sdk()
    sdk.stop_limit_order_gtc_sell.side_effect = RuntimeError("503")
    with pytest.raises(UnprotectedPositionError, match="POSITION IS UNPROTECTED"):
        CoinbaseBroker(_settings(), client=sdk).submit_order(_order())


def test_stop_rounds_up_and_limit_floor_rounds_down():
    sdk = _sdk()
    CoinbaseBroker(_settings(), client=sdk).submit_order(_order(stop="94000.123"))
    stop_kwargs = sdk.stop_limit_order_gtc_sell.call_args.kwargs
    assert stop_kwargs["stop_price"] == "94000.13"
    assert stop_kwargs["limit_price"] == "93530.12"


def test_quantity_below_base_min_size_unfillable():
    """Above base_increment but below base_min_size must fail typed and local,
    not as a raw venue rejection."""
    sdk = _sdk()
    sdk.get_product.return_value = SimpleNamespace(
        price="97000.00",
        base_increment="0.00000001",
        quote_increment="0.01",
        base_min_size="0.001",
    )
    with pytest.raises(UnfillableQuantityError, match="below venue minimum"):
        CoinbaseBroker(_settings(), client=sdk).submit_order(_order(quantity="0.0004"))


def test_venue_rejection_raises_broker_error():
    sdk = _sdk()
    sdk.market_order_buy.return_value = SimpleNamespace(
        success=False, error_response={"error": "INSUFFICIENT_FUND"}
    )
    with pytest.raises(BrokerError, match="rejected order"):
        CoinbaseBroker(_settings(), client=sdk).submit_order(_order())


# --- sells: cancel, await release, then close -----------------------------------


def test_sell_cancels_awaits_release_then_market_sells():
    """Cancel is async — the sell must wait until the venue reports no open orders
    (holds released) before selling."""
    sdk = _sdk()
    sdk.list_orders.side_effect = [
        SimpleNamespace(orders=[SimpleNamespace(order_id="stop-1")], has_next=False),  # cancel scan
        SimpleNamespace(orders=[SimpleNamespace(order_id="stop-1")], has_next=False),  # still open
        SimpleNamespace(orders=[], has_next=False),  # released
    ]
    sdk.cancel_orders.return_value = SimpleNamespace(
        results=[SimpleNamespace(success=True, order_id="stop-1")]
    )
    result = CoinbaseBroker(_settings(), client=sdk).submit_order(
        _order(side="sell", stop=None, quantity="0.5"), client_order_id="cycle2-BTC-sell"
    )
    sdk.cancel_orders.assert_called_once_with(order_ids=["stop-1"])
    assert sdk.list_orders.call_count == 3
    sell_kwargs = sdk.market_order_sell.call_args.kwargs
    assert sell_kwargs["base_size"] == "0.50000000"
    assert sell_kwargs["client_order_id"] == "cycle2-BTC-sell"  # idempotent sells
    assert result.client_order_id == "cycle2-BTC-sell"


def test_sell_aborts_when_holds_never_release():
    sdk = _sdk()
    sdk.list_orders.return_value = SimpleNamespace(
        orders=[SimpleNamespace(order_id="stop-1")], has_next=False
    )
    sdk.cancel_orders.return_value = SimpleNamespace(
        results=[SimpleNamespace(success=True, order_id="stop-1")]
    )
    with pytest.raises(BrokerError, match="did not clear"):
        CoinbaseBroker(_settings(), client=sdk).submit_order(
            _order(side="sell", stop=None, quantity="0.5")
        )
    sdk.market_order_sell.assert_not_called()


def test_sell_aborts_when_cancel_results_missing_or_partial():
    """Copilot finding: an empty/partial results list for requested ids must
    abort — a missing row means an order may still hold the funds."""
    sdk = _sdk()
    sdk.list_orders.return_value = SimpleNamespace(
        orders=[SimpleNamespace(order_id="stop-1"), SimpleNamespace(order_id="stop-2")],
        has_next=False,
    )
    sdk.cancel_orders.return_value = SimpleNamespace(
        results=[SimpleNamespace(success=True, order_id="stop-1")]  # stop-2 missing
    )
    with pytest.raises(BrokerError, match="did not confirm cancellation"):
        CoinbaseBroker(_settings(), client=sdk).submit_order(
            _order(side="sell", stop=None, quantity="0.5")
        )
    sdk.market_order_sell.assert_not_called()


def test_sell_aborts_when_cancellation_reported_failed():
    sdk = _sdk()
    sdk.list_orders.return_value = SimpleNamespace(
        orders=[SimpleNamespace(order_id="stop-1")], has_next=False
    )
    sdk.cancel_orders.return_value = SimpleNamespace(
        results=[SimpleNamespace(success=False, order_id="stop-1")]
    )
    with pytest.raises(BrokerError, match="did not confirm cancellation"):
        CoinbaseBroker(_settings(), client=sdk).submit_order(
            _order(side="sell", stop=None, quantity="0.5")
        )
    sdk.market_order_sell.assert_not_called()


def test_sell_with_no_open_orders_skips_wait():
    sdk = _sdk()
    CoinbaseBroker(_settings(), client=sdk).submit_order(
        _order(side="sell", stop=None, quantity="0.12345678")
    )
    assert sdk.market_order_sell.call_args.kwargs["base_size"] == "0.12345678"


def test_sell_with_stop_rejected_as_malformed():
    with pytest.raises(BrokerError, match="must not carry a stop"):
        CoinbaseBroker(_settings(), client=_sdk()).submit_order(_order(side="sell", stop="94000"))


# --- cancel-all -------------------------------------------------------------------


def test_cancel_all_counts_only_acknowledged():
    sdk = _sdk()
    sdk.list_orders.return_value = SimpleNamespace(
        orders=[SimpleNamespace(order_id="a"), SimpleNamespace(order_id="b")], has_next=False
    )
    sdk.cancel_orders.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(success=True, order_id="a"),
            SimpleNamespace(success=False, order_id="b", failure_reason="UNKNOWN"),
        ]
    )
    assert CoinbaseBroker(_settings(), client=sdk).cancel_all_orders() == 1


def test_open_orders_follow_pagination():
    """Open orders beyond page one must still be cancelled."""
    sdk = _sdk()
    sdk.list_orders.side_effect = [
        SimpleNamespace(orders=[SimpleNamespace(order_id="a")], has_next=True, cursor="c1"),
        SimpleNamespace(orders=[SimpleNamespace(order_id="b")], has_next=False),
    ]
    sdk.cancel_orders.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(success=True, order_id="a"),
            SimpleNamespace(success=True, order_id="b"),
        ]
    )
    assert CoinbaseBroker(_settings(), client=sdk).cancel_all_orders() == 2
    sdk.cancel_orders.assert_called_once_with(order_ids=["a", "b"])


# --- balances -> positions / account -----------------------------------------------


def _accounts_sdk() -> MagicMock:
    sdk = _sdk()
    sdk.get_accounts.return_value = SimpleNamespace(
        accounts=[
            SimpleNamespace(
                currency="USD", available_balance={"value": "40000.25"}, hold={"value": "0"}
            ),
            # The designed steady state: the ENTIRE position is held by its stop.
            SimpleNamespace(
                currency="BTC", available_balance={"value": "0"}, hold={"value": "0.5"}
            ),
            SimpleNamespace(currency="ETH", available_balance={"value": "0"}, hold={"value": "0"}),
        ],
        has_next=False,
    )
    return sdk


def test_positions_count_held_funds():
    """A stop-protected position lives in `hold` — it must not vanish from get_positions."""
    (pos,) = CoinbaseBroker(_settings(), client=_accounts_sdk()).get_positions()
    assert pos.symbol == "BTC-USD"
    assert pos.quantity == Decimal("0.5")
    assert pos.avg_entry_price is None


def test_account_equity_counts_held_funds():
    """Equity must not collapse to cash when stops are working."""
    summary = CoinbaseBroker(_settings(), client=_accounts_sdk()).get_account()
    assert summary.cash == Decimal("40000.25")
    assert summary.equity == Decimal("88500.25")  # cash + 0.5 BTC x 97000


def test_held_quote_funds_are_equity_but_not_cash():
    """Copilot finding: quote funds reserved by open orders are not deployable —
    counting them as cash would let the gate size against unspendable money."""
    sdk = _sdk()
    sdk.get_accounts.return_value = SimpleNamespace(
        accounts=[
            SimpleNamespace(
                currency="USD",
                available_balance={"value": "30000"},
                hold={"value": "10000"},  # committed to an open order
            )
        ],
        has_next=False,
    )
    summary = CoinbaseBroker(_settings(), client=sdk).get_account()
    assert summary.cash == Decimal("30000")  # available only
    assert summary.equity == Decimal("40000")  # held funds are still ours


def test_unpriceable_holding_excluded_not_fatal():
    """Delisted/staked dust must not break the account read."""
    sdk = _accounts_sdk()
    sdk.get_accounts.return_value.accounts.append(
        SimpleNamespace(currency="ETH2", available_balance={"value": "1.0"}, hold={"value": "0"})
    )

    def product(symbol):
        if symbol == "ETH2-USD":
            raise RuntimeError("404 product not found")
        return SimpleNamespace(price="97000.00")

    sdk.get_product.side_effect = product
    summary = CoinbaseBroker(_settings(), client=sdk).get_account()
    assert summary.equity == Decimal("88500.25")  # ETH2 excluded, read survives


def test_accounts_follow_pagination():
    sdk = _sdk()
    sdk.get_accounts.side_effect = [
        SimpleNamespace(
            accounts=[
                SimpleNamespace(
                    currency="USD", available_balance={"value": "100"}, hold={"value": "0"}
                )
            ],
            has_next=True,
            cursor="c1",
        ),
        SimpleNamespace(
            accounts=[
                SimpleNamespace(
                    currency="BTC", available_balance={"value": "1"}, hold={"value": "0"}
                )
            ],
            has_next=False,
        ),
    ]
    positions = CoinbaseBroker(_settings(), client=sdk).get_positions()
    assert [p.symbol for p in positions] == ["BTC-USD"]


def test_declares_venue_name_and_asset_class():
    broker = CoinbaseBroker(_settings(), client=_sdk())
    assert (broker.name, broker.asset_class) == ("coinbase", "crypto")


def test_buy_returns_the_protecting_stop_id():
    submitted = CoinbaseBroker(_settings(), client=_sdk()).submit_order(_order())
    assert submitted.broker_order_id == "entry-1"
    assert submitted.protective_order_id == "stop-1"


def test_unprotected_error_carries_the_filled_entry_id():
    """The executor must still BOOK the naked position's entry fill."""
    sdk = _sdk()
    sdk.stop_limit_order_gtc_sell.side_effect = RuntimeError("503")
    with pytest.raises(UnprotectedPositionError) as excinfo:
        CoinbaseBroker(_settings(), client=sdk).submit_order(_order())
    assert excinfo.value.broker_order_id == "entry-1"


def test_get_order_maps_filled_state_with_fees_and_time():
    from datetime import UTC, datetime

    sdk = _sdk()
    sdk.get_order.return_value = SimpleNamespace(
        order=SimpleNamespace(
            order_id="stop-1",
            status="FILLED",
            filled_size="0.5",
            average_filled_price="93950.10",
            last_fill_time="2026-09-15T10:00:00.123Z",
            total_fees="1.25",
        )
    )
    venue = CoinbaseBroker(_settings(), client=sdk).get_order("stop-1")
    assert venue.terminal and venue.status == "filled"
    assert venue.filled_quantity == Decimal("0.5")
    assert venue.fill_price == Decimal("93950.10")
    assert venue.filled_at == datetime(2026, 9, 15, 10, 0, 0, 123000, tzinfo=UTC)
    assert venue.fees == Decimal("1.25")


def test_get_order_maps_open_and_cancelled_states():
    sdk = _sdk()
    sdk.get_order.return_value = SimpleNamespace(
        order=SimpleNamespace(order_id="s", status="OPEN", filled_size="0")
    )
    assert not CoinbaseBroker(_settings(), client=sdk).get_order("s").terminal
    sdk.get_order.return_value = SimpleNamespace(
        order=SimpleNamespace(order_id="s", status="CANCELLED", filled_size="0")
    )
    cancelled = CoinbaseBroker(_settings(), client=sdk).get_order("s")
    assert cancelled.terminal and cancelled.filled_quantity == 0 and cancelled.fill_price is None


def test_get_order_wraps_sdk_failures():
    sdk = _sdk()
    sdk.get_order.side_effect = RuntimeError("504")
    with pytest.raises(BrokerError, match="get_order failed"):
        CoinbaseBroker(_settings(), client=sdk).get_order("s")


def test_get_order_404_is_an_unknown_order_error():
    class NotFound(Exception):
        response = SimpleNamespace(status_code=404)

    sdk = _sdk()
    sdk.get_order.side_effect = NotFound("not found")
    with pytest.raises(UnknownOrderError):
        CoinbaseBroker(_settings(), client=sdk).get_order("gone")


# --- standalone protective stop ---------------------------------------------------


def test_place_stop_is_the_same_stop_limit_shape_as_after_a_buy():
    sdk = _sdk()
    stop_id = CoinbaseBroker(_settings(), client=sdk).place_stop(
        "BTC-USD", Decimal("0.123456789"), Decimal("94000.004"), "c1:BTC-USD:protect"
    )
    assert stop_id == "stop-1"
    kwargs = sdk.stop_limit_order_gtc_sell.call_args.kwargs
    assert kwargs["base_size"] == "0.12345678"  # rounded DOWN to the base increment
    assert kwargs["stop_price"] == "94000.01"  # rounded UP: tighter
    assert Decimal(kwargs["limit_price"]) < Decimal(kwargs["stop_price"])
    assert kwargs["stop_direction"] == "STOP_DIRECTION_STOP_DOWN"
    assert kwargs["client_order_id"] == "c1:BTC-USD:protect"


def test_place_stop_failures_are_broker_errors():
    sdk = _sdk()
    sdk.stop_limit_order_gtc_sell.side_effect = RuntimeError("503")
    with pytest.raises(BrokerError):
        CoinbaseBroker(_settings(), client=sdk).place_stop("BTC-USD", Decimal(1), Decimal(90), "k")
    with pytest.raises(UnfillableQuantityError):
        CoinbaseBroker(_settings(), client=_sdk()).place_stop(
            "BTC-USD", Decimal("0.000000001"), Decimal(90), "k"
        )
