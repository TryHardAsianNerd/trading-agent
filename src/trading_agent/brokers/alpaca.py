"""Alpaca broker for US stocks, paper endpoints by default.

Buys are fractional DAY market orders with no stop attached (Alpaca takes fractions only
on simple DAY orders); the app holds the stop and sells when a cycle's price reaches it.
Sells are full closes. Alpaca rejects a reused client_order_id instead of returning the
original, so a failed submit or close is replayed from the venue's own record.
"""

import uuid
from decimal import ROUND_DOWN, ROUND_UP, Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Order
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopOrderRequest,
)

from trading_agent.brokers.base import (
    AccountSummary,
    BrokerError,
    BrokerPosition,
    SubmittedOrder,
    UnfillableQuantityError,
    UnknownOrderError,
    VenueOrder,
    WrongVenueError,
)
from trading_agent.config import LiveTradingNotConfirmedError, Settings
from trading_agent.graph.state import ApprovedOrder
from trading_agent.log import get_logger

log = get_logger(__name__)

ASSET_CLASS = "stock"
FRACTIONAL_STEP = Decimal("0.000000001")
# Statuses after which an order can never fill further; done_for_day is NOT terminal for GTC.
_TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}


class AlpacaBroker:
    """Broker implementation for US stocks via alpaca-py (paper by default)."""

    def __init__(self, settings: Settings, client: TradingClient | None = None) -> None:
        """Build the broker; LiveTradingNotConfirmedError for unconfirmed live mode.

        The guard tests the same is_paper predicate that picks the endpoint, so no settings
        object can reach the live URL unconfirmed. ValueError if the keys are missing.
        """
        if not settings.is_paper and settings.live_trading_confirmed != "yes":
            raise LiveTradingNotConfirmedError()
        if client is None:
            if not settings.alpaca_api_key or not settings.alpaca_secret_key:
                raise ValueError("AlpacaBroker requires ALPACA_API_KEY and ALPACA_SECRET_KEY")
            client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=settings.is_paper,
            )
        self._client = client
        self._paper = settings.is_paper
        self.name = "alpaca"
        self.asset_class = ASSET_CLASS

    def submit_order(
        self, order: ApprovedOrder, client_order_id: str | None = None
    ) -> SubmittedOrder:
        """Submit a gate-approved stock order; a buy without a client_order_id gets a new one."""
        if order.asset_class != ASSET_CLASS:
            raise WrongVenueError(
                f"AlpacaBroker trades {ASSET_CLASS!r}, got {order.asset_class!r} for {order.symbol}"
            )
        if order.side == "sell":
            if order.stop_loss_price is not None:
                raise BrokerError(
                    f"{order.symbol}: sell orders must not carry a stop price "
                    "(sells are full closes; a sell-side stop child is nonsense)"
                )
            return self._close_position(order)
        return self._submit_buy(order, client_order_id or f"ta-{uuid.uuid4()}")

    def _submit_buy(self, order: ApprovedOrder, client_order_id: str) -> SubmittedOrder:
        """Submit a fractional DAY market buy, quantity rounded down to 9 decimal places."""
        qty = order.quantity.quantize(FRACTIONAL_STEP, rounding=ROUND_DOWN)
        if qty <= 0:
            raise UnfillableQuantityError(f"{order.symbol}: quantity {order.quantity} rounds to 0")

        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=float(qty),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            result = self._client.submit_order(request)
        except Exception as e:
            # Lost response / reused key (422): the venue's FIRST submission is the order.
            result = self._order_by_client_id(client_order_id)
            if result is None:
                raise BrokerError(f"Alpaca submit_order failed for {order.symbol}: {e}") from e
            log.warning(
                "order_replayed_from_venue",
                broker="alpaca",
                symbol=order.symbol,
                client_order_id=client_order_id,
                broker_order_id=str(result.id),
            )

        submitted = SubmittedOrder(
            broker_order_id=str(result.id),
            status=str(getattr(result.status, "value", result.status)),
            submitted_quantity=qty,
            client_order_id=client_order_id,
        )
        log.info(
            "order_submitted",
            broker="alpaca",
            paper=self._paper,
            symbol=order.symbol,
            side="buy",
            quantity=str(qty),
            app_held_stop=None if order.stop_loss_price is None else str(order.stop_loss_price),
            client_order_id=client_order_id,
            broker_order_id=submitted.broker_order_id,
            status=submitted.status,
        )
        return submitted

    def place_stop(
        self, symbol: str, quantity: Decimal, stop_price: Decimal, client_order_id: str
    ) -> str:
        """Place a GTC stop sell on the whole shares of a held position and return its id.

        Alpaca stop orders take whole shares only, so a fractional remainder stays unprotected
        (and is logged).
        """
        qty = quantity.to_integral_value(rounding=ROUND_DOWN)
        if qty <= 0:
            raise UnfillableQuantityError(f"{symbol}: {quantity} rounds to 0 whole shares")
        if qty != quantity:
            log.warning("stop_fractional_remainder_unprotected", symbol=symbol, qty=str(quantity))
        request = StopOrderRequest(
            symbol=symbol,
            qty=int(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=_stop_tick(stop_price),
            client_order_id=client_order_id,
        )
        try:
            result = self._client.submit_order(request)
        except Exception as e:
            result = self._order_by_client_id(client_order_id)
            if result is None:
                raise BrokerError(f"Alpaca stop order failed for {symbol}: {e}") from e
        log.info(
            "stop_placed",
            broker="alpaca",
            symbol=symbol,
            quantity=str(qty),
            stop_price=str(stop_price),
            broker_order_id=str(result.id),
        )
        return str(result.id)

    def _close_position(self, order: ApprovedOrder) -> SubmittedOrder:
        """Cancel the symbol's open orders, then close the entire position venue-side.

        The cancel releases shares a working stop reserves (otherwise Alpaca rejects the sell);
        close-position sells fractional shares too, leaving no dust.
        """
        cancelled = self._cancel_symbol_orders(order.symbol)
        try:
            result = self._client.close_position(order.symbol)
        except Exception as e:
            # An already-succeeded close leaves no position; replay the venue's latest filled sell.
            result = self._replay_close(order.symbol)
            if result is None:
                raise BrokerError(f"Alpaca close_position failed for {order.symbol}: {e}") from e
            log.warning("close_replayed_from_venue", broker="alpaca", symbol=order.symbol)

        closed_qty = Decimal(str(result.qty)) if getattr(result, "qty", None) else order.quantity
        submitted = SubmittedOrder(
            broker_order_id=str(result.id),
            status=str(getattr(result.status, "value", result.status)),
            submitted_quantity=closed_qty,
            client_order_id=None,
        )
        log.info(
            "position_closed",
            broker="alpaca",
            paper=self._paper,
            symbol=order.symbol,
            quantity=str(closed_qty),
            symbol_orders_cancelled=cancelled,
            broker_order_id=submitted.broker_order_id,
            status=submitted.status,
        )
        return submitted

    def _order_by_client_id(self, client_order_id: str) -> Order | None:
        """The venue order carrying an idempotency key; None if none exists or the lookup fails."""
        try:
            return self._client.get_order_by_client_id(client_order_id)
        except Exception:
            return None

    def _replay_close(self, symbol: str) -> Order | None:
        """The latest filled sell for a symbol once its position is gone; None while it is held."""
        try:
            self._client.get_open_position(symbol)
            return None
        except Exception:
            pass
        try:
            closed = self._client.get_orders(
                GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED, symbols=[symbol], side=OrderSide.SELL, limit=5
                )
            )
        except Exception:
            return None
        for venue_order in closed:
            if venue_order.filled_qty and Decimal(str(venue_order.filled_qty)) > 0:
                return venue_order
        return None

    def _cancel_symbol_orders(self, symbol: str) -> int:
        """Cancel a symbol's open orders and return the count; a failure raises, halting a sell."""
        try:
            open_orders = self._client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            )
            for open_order in open_orders:
                self._client.cancel_order_by_id(open_order.id)
        except Exception as e:
            raise BrokerError(f"Alpaca cancel of {symbol} open orders failed: {e}") from e
        return len(open_orders)

    def get_order(self, broker_order_id: str) -> VenueOrder:
        """Look up one order for fill reconciliation; fees are 0 because Alpaca charges none."""
        try:
            venue = self._client.get_order_by_id(broker_order_id)
        except Exception as e:
            if getattr(e, "status_code", None) == 404:
                raise UnknownOrderError(f"Alpaca has no order {broker_order_id}") from e
            raise BrokerError(f"Alpaca get_order_by_id failed for {broker_order_id}: {e}") from e
        status = str(getattr(venue.status, "value", venue.status)).lower()
        filled_qty = Decimal(str(venue.filled_qty)) if venue.filled_qty else Decimal(0)
        price = getattr(venue, "filled_avg_price", None)
        return VenueOrder(
            status=status,
            terminal=status in _TERMINAL_STATUSES,
            filled_quantity=filled_qty,
            fill_price=Decimal(str(price)) if price and filled_qty > 0 else None,
            filled_at=getattr(venue, "filled_at", None) if filled_qty > 0 else None,
        )

    def cancel_all_orders(self) -> int:
        """Cancel every open order and return how many the venue confirmed; failures are logged."""
        try:
            responses = self._client.cancel_orders()
        except Exception as e:
            raise BrokerError(f"Alpaca cancel_orders failed: {e}") from e

        confirmed = 0
        for r in responses:
            status = getattr(r, "status", None)
            if status is not None and status < 400:
                confirmed += 1
            else:
                log.error(
                    "order_cancel_failed",
                    broker="alpaca",
                    order_id=str(getattr(r, "id", "?")),
                    http_status=status,
                )
        log.warning(
            "all_orders_cancelled",
            broker="alpaca",
            confirmed=confirmed,
            attempted=len(responses),
        )
        return confirmed

    def get_positions(self) -> list[BrokerPosition]:
        """Every open stock position."""
        try:
            positions = self._client.get_all_positions()
        except Exception as e:
            raise BrokerError(f"Alpaca get_all_positions failed: {e}") from e
        return [
            BrokerPosition(
                symbol=p.symbol,
                asset_class=ASSET_CLASS,
                quantity=Decimal(str(p.qty)),
                avg_entry_price=Decimal(str(p.avg_entry_price)),
            )
            for p in positions
        ]

    def get_account(self) -> AccountSummary:
        """Account equity and cash; BrokerError if the response lacks either."""
        try:
            account = self._client.get_account()
        except Exception as e:
            raise BrokerError(f"Alpaca get_account failed: {e}") from e
        if account.equity is None or account.cash is None:
            raise BrokerError("Alpaca account response missing equity/cash")
        return AccountSummary(equity=Decimal(str(account.equity)), cash=Decimal(str(account.cash)))


def _stop_tick(value: Decimal) -> float:
    """Round a long stop price up to Alpaca's tick ($0.01, or $0.0001 below $1).

    Up tightens the stop; rounding down would widen the loss past what the risk engine sized.
    """
    tick = Decimal("0.01") if value >= 1 else Decimal("0.0001")
    return float(value.quantize(tick, rounding=ROUND_UP))
