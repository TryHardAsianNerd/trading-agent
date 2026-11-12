"""Coinbase Advanced Trade broker for crypto, sandbox by default.

Buys are a market IOC entry, polled to a terminal status, then a GTC stop-limit sell for
the filled size; a gap through the limit floor can leave that exit unfilled. Sells are
full closes. Funds committed to open orders sit in `hold`, so positions and equity count
it too. Known limitation: the sandbox serves only the Accounts and Orders endpoints
(no get_product) with canned data, so a paper submit or account read cannot complete.
"""

import time
import uuid
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal

from coinbase.rest import RESTClient

from trading_agent.brokers.base import (
    AccountSummary,
    BrokerError,
    BrokerPosition,
    SubmittedOrder,
    UnfillableQuantityError,
    UnknownOrderError,
    UnprotectedPositionError,
    VenueOrder,
    WrongVenueError,
)
from trading_agent.config import LiveTradingNotConfirmedError, Settings
from trading_agent.graph.state import ApprovedOrder
from trading_agent.log import get_logger

log = get_logger(__name__)

ASSET_CLASS = "crypto"
SANDBOX_BASE_URL = "api-sandbox.coinbase.com"
QUOTE_CURRENCIES = ("USD", "USDC")
# 0.5% under the trigger bounds the worst fill; a gap through the floor goes unfilled.
STOP_LIMIT_BUFFER = Decimal("0.005")
_TERMINAL_STATUSES = {"FILLED", "CANCELLED", "EXPIRED", "FAILED"}
# Poll budget for venue state to settle (IOC fills and cancels are fast).
_POLL_ATTEMPTS = 10
_POLL_DELAY_SECONDS = 0.3


class CoinbaseBroker:
    """Broker implementation for crypto via coinbase-advanced-py (sandbox by default)."""

    def __init__(self, settings: Settings, client: RESTClient | None = None) -> None:
        """Build the broker; LiveTradingNotConfirmedError for unconfirmed live mode.

        The guard tests the same is_paper predicate that picks the sandbox URL. ValueError if the
        keys are missing.
        """
        if not settings.is_paper and settings.live_trading_confirmed != "yes":
            raise LiveTradingNotConfirmedError()
        if client is None:
            if not settings.coinbase_api_key or not settings.coinbase_api_secret:
                raise ValueError("CoinbaseBroker requires COINBASE_API_KEY and COINBASE_API_SECRET")
            kwargs = {"base_url": SANDBOX_BASE_URL} if settings.is_paper else {}
            client = RESTClient(
                api_key=settings.coinbase_api_key,
                api_secret=settings.coinbase_api_secret,
                **kwargs,
            )
        self._client = client
        self._paper = settings.is_paper
        self._products: dict[str, tuple[Decimal, Decimal, Decimal | None]] = {}
        self.name = "coinbase"
        self.asset_class = ASSET_CLASS

    def submit_order(
        self, order: ApprovedOrder, client_order_id: str | None = None
    ) -> SubmittedOrder:
        """Submit a gate-approved crypto order; a reused client_order_id returns the original.

        Raises UnprotectedPositionError when a buy filled but its stop could not be placed: the
        position is live and unprotected, and the entry must never be retried blindly.
        """
        if order.asset_class != ASSET_CLASS:
            raise WrongVenueError(
                f"CoinbaseBroker trades {ASSET_CLASS!r}, got {order.asset_class!r} "
                f"for {order.symbol}"
            )
        coi = client_order_id or f"ta-{uuid.uuid4()}"
        if order.side == "sell":
            if order.stop_loss_price is not None:
                raise BrokerError(
                    f"{order.symbol}: sell orders must not carry a stop price "
                    "(sells are full closes)"
                )
            return self._close_position(order, coi)
        return self._submit_buy(order, coi)

    def _submit_buy(self, order: ApprovedOrder, client_order_id: str) -> SubmittedOrder:
        """Market-buy, wait for the terminal fill, then place the stop as "<client_order_id>-stop".

        An entry still not terminal after the poll budget is a BrokerError, never "no fill".
        """
        base_inc, quote_inc, base_min = self._product(order.symbol)
        qty = order.quantity.quantize(base_inc, rounding=ROUND_DOWN)
        if qty <= 0 or (base_min is not None and qty < base_min):
            raise UnfillableQuantityError(
                f"{order.symbol}: quantity {order.quantity} below venue minimum "
                f"(base_increment {base_inc}, base_min_size {base_min})"
            )

        try:
            # base_size is valid for IOC buys; SDK docstring (quote_size) is stale — don't fix it.
            response = self._client.market_order_buy(
                client_order_id=client_order_id,
                product_id=order.symbol,
                base_size=str(qty),
            )
        except Exception as e:
            raise BrokerError(f"Coinbase market_order_buy failed for {order.symbol}: {e}") from e
        entry_id = _order_id_from(response, order.symbol)

        status, filled = self._await_terminal(entry_id, order.symbol)
        stop_sent = limit_sent = stop_id = None
        if order.stop_loss_price is not None and filled > 0:
            stop_sent = str(order.stop_loss_price.quantize(quote_inc, rounding=ROUND_UP))
            limit_sent = str(
                (order.stop_loss_price * (1 - STOP_LIMIT_BUFFER)).quantize(
                    quote_inc, rounding=ROUND_DOWN
                )
            )
            try:
                stop_response = self._client.stop_limit_order_gtc_sell(
                    client_order_id=f"{client_order_id}-stop",
                    product_id=order.symbol,
                    base_size=str(filled),
                    limit_price=limit_sent,
                    stop_price=stop_sent,
                    stop_direction="STOP_DIRECTION_STOP_DOWN",
                )
                stop_id = _order_id_from(stop_response, order.symbol)
            except Exception as e:
                log.error(
                    "position_unprotected",
                    broker="coinbase",
                    symbol=order.symbol,
                    filled=str(filled),
                    entry_order_id=entry_id,
                )
                raise UnprotectedPositionError(
                    f"Coinbase buy FILLED ({filled} {order.symbol}) but the protecting "
                    f"stop could not be placed — POSITION IS UNPROTECTED: {e}",
                    broker_order_id=entry_id,
                ) from e

        submitted = SubmittedOrder(
            broker_order_id=entry_id,
            status=status,
            submitted_quantity=qty,
            filled_quantity=filled,
            client_order_id=client_order_id,
            protective_order_id=stop_id,
        )
        log.info(
            "order_submitted",
            broker="coinbase",
            paper=self._paper,
            symbol=order.symbol,
            side="buy",
            quantity=str(qty),
            filled=str(filled),
            venue_status=status,
            stop_loss_sent=stop_sent,
            stop_limit_floor_sent=limit_sent,
            client_order_id=client_order_id,
            broker_order_id=entry_id,
        )
        return submitted

    def place_stop(
        self, symbol: str, quantity: Decimal, stop_price: Decimal, client_order_id: str
    ) -> str:
        """Place a buy-style GTC stop-limit sell on a held position and return its id."""
        base_inc, quote_inc, _ = self._product(symbol)
        qty = quantity.quantize(base_inc, rounding=ROUND_DOWN)
        if qty <= 0:
            raise UnfillableQuantityError(f"{symbol}: {quantity} rounds to 0 at {base_inc}")
        try:
            response = self._client.stop_limit_order_gtc_sell(
                client_order_id=client_order_id,
                product_id=symbol,
                base_size=str(qty),
                limit_price=str(
                    (stop_price * (1 - STOP_LIMIT_BUFFER)).quantize(quote_inc, rounding=ROUND_DOWN)
                ),
                stop_price=str(stop_price.quantize(quote_inc, rounding=ROUND_UP)),
                stop_direction="STOP_DIRECTION_STOP_DOWN",
            )
        except Exception as e:
            raise BrokerError(f"Coinbase stop order failed for {symbol}: {e}") from e
        return _order_id_from(response, symbol)

    def _close_position(self, order: ApprovedOrder, client_order_id: str) -> SubmittedOrder:
        """Cancel the product's open orders, wait for their holds to release, then market-sell."""
        base_inc, _, _ = self._product(order.symbol)
        qty = order.quantity.quantize(base_inc, rounding=ROUND_DOWN)
        if qty <= 0:
            raise UnfillableQuantityError(
                f"{order.symbol}: position {order.quantity} quantizes to 0 "
                f"(base_increment {base_inc})"
            )
        cancelled = self._cancel_product_orders(order.symbol)
        if cancelled:
            self._await_no_open_orders(order.symbol)
        try:
            response = self._client.market_order_sell(
                client_order_id=client_order_id,
                product_id=order.symbol,
                base_size=str(qty),
            )
        except Exception as e:
            raise BrokerError(f"Coinbase market_order_sell failed for {order.symbol}: {e}") from e
        sell_id = _order_id_from(response, order.symbol)

        submitted = SubmittedOrder(
            broker_order_id=sell_id,
            status="submitted",
            submitted_quantity=qty,
            client_order_id=client_order_id,
        )
        log.info(
            "position_closed",
            broker="coinbase",
            paper=self._paper,
            symbol=order.symbol,
            quantity=str(qty),
            product_orders_cancelled=cancelled,
            client_order_id=client_order_id,
            broker_order_id=sell_id,
        )
        return submitted

    def get_order(self, broker_order_id: str) -> VenueOrder:
        """Look up one order for fill reconciliation, with fees from the venue's total_fees."""
        try:
            venue = self._client.get_order(broker_order_id).order
        except Exception as e:
            response = getattr(e, "response", None)
            if getattr(response, "status_code", None) == 404:
                raise UnknownOrderError(f"Coinbase has no order {broker_order_id}") from e
            raise BrokerError(f"Coinbase get_order failed for {broker_order_id}: {e}") from e
        status = str(getattr(venue, "status", "")).upper()
        filled = Decimal(str(getattr(venue, "filled_size", None) or "0"))
        price = getattr(venue, "average_filled_price", None)
        fill_time = getattr(venue, "last_fill_time", None)
        return VenueOrder(
            status=status.lower(),
            terminal=status in _TERMINAL_STATUSES,
            filled_quantity=filled,
            fill_price=Decimal(str(price)) if price and filled > 0 else None,
            filled_at=_parse_ts(fill_time) if fill_time and filled > 0 else None,
            fees=Decimal(str(getattr(venue, "total_fees", None) or "0")),
        )

    def _await_terminal(self, order_id: str, symbol: str) -> tuple[str, Decimal]:
        """Poll an order to a terminal status and return (status, filled base size).

        An instant read can show 0 or a partial fill while the fill settles, which would skip or
        undersize the stop. BrokerError if it stays unknown within the poll budget.
        """
        last_error: Exception | None = None
        for attempt in range(_POLL_ATTEMPTS):
            try:
                venue_order = self._client.get_order(order_id).order
                status = str(getattr(venue_order, "status", "")).upper()
                if status in _TERMINAL_STATUSES:
                    return status, Decimal(str(venue_order.filled_size or "0"))
                last_error = None
            except Exception as e:  # order may not be queryable yet — retry
                last_error = e
            if attempt < _POLL_ATTEMPTS - 1:
                time.sleep(_POLL_DELAY_SECONDS)
        raise BrokerError(
            f"Coinbase order {order_id} for {symbol} did not reach a terminal status "
            f"within the poll budget — state UNKNOWN, treat as failed and investigate"
            + (f" (last error: {last_error})" if last_error else "")
        )

    def _await_no_open_orders(self, symbol: str) -> None:
        """Wait until a product has no open orders; batch cancel is asynchronous."""
        for attempt in range(_POLL_ATTEMPTS):
            if not self._open_orders(symbol):
                return
            if attempt < _POLL_ATTEMPTS - 1:
                time.sleep(_POLL_DELAY_SECONDS)
        raise BrokerError(
            f"Coinbase open orders for {symbol} did not clear after cancellation — "
            "funds may still be held; not proceeding with the sell"
        )

    def _open_orders(self, symbol: str | None = None) -> list:
        """Every open order, for one product or all, across pages."""
        orders: list = []
        cursor = None
        try:
            while True:
                kwargs = {"order_status": ["OPEN"]}
                if symbol is not None:
                    kwargs["product_ids"] = [symbol]
                if cursor:
                    kwargs["cursor"] = cursor
                page = self._client.list_orders(**kwargs)
                orders.extend(getattr(page, "orders", None) or [])
                if not getattr(page, "has_next", False):
                    return orders
                cursor = getattr(page, "cursor", None)
        except Exception as e:
            raise BrokerError(f"Coinbase list_orders failed: {e}") from e

    def _cancel_product_orders(self, symbol: str) -> int:
        """Cancel a product's open orders; BrokerError unless the venue confirms every one.

        Returns the acknowledged count; the cancels complete asynchronously.
        """
        open_orders = self._open_orders(symbol)
        if not open_orders:
            return 0
        requested = {str(o.order_id) for o in open_orders}
        try:
            result = self._client.cancel_orders(order_ids=list(requested))
        except Exception as e:
            raise BrokerError(f"Coinbase cancel of {symbol} open orders failed: {e}") from e
        results = getattr(result, "results", None) or []
        succeeded = {
            str(getattr(r, "order_id", "")) for r in results if getattr(r, "success", False)
        }
        missing = requested - succeeded
        if missing:
            raise BrokerError(
                f"Coinbase did not confirm cancellation of {len(missing)} order(s) for "
                f"{symbol} — funds may still be held; not proceeding"
            )
        return len(succeeded)

    def cancel_all_orders(self) -> int:
        """Cancel all open orders and return how many the venue acknowledged; logs refusals."""
        open_orders = self._open_orders()
        if not open_orders:
            return 0
        try:
            result = self._client.cancel_orders(order_ids=[o.order_id for o in open_orders])
        except Exception as e:
            raise BrokerError(f"Coinbase cancel_orders failed: {e}") from e

        acknowledged = 0
        for r in getattr(result, "results", None) or []:
            if getattr(r, "success", False):
                acknowledged += 1
            else:
                log.error(
                    "order_cancel_failed",
                    broker="coinbase",
                    order_id=str(getattr(r, "order_id", "?")),
                    reason=str(getattr(r, "failure_reason", "?")),
                )
        log.warning(
            "all_orders_cancelled",
            broker="coinbase",
            acknowledged=acknowledged,
            attempted=len(open_orders),
        )
        return acknowledged

    def get_positions(self) -> list[BrokerPosition]:
        """Every non-quote holding as a "<CURRENCY>-USD" position, available plus held funds."""
        positions = []
        for account in self._accounts():
            currency = getattr(account, "currency", "")
            total = _amount(getattr(account, "available_balance", None)) + _amount(
                getattr(account, "hold", None)
            )
            if currency in QUOTE_CURRENCIES or total <= 0:
                continue
            positions.append(
                BrokerPosition(
                    symbol=f"{currency}-USD",
                    asset_class=ASSET_CLASS,
                    quantity=total,
                    avg_entry_price=None,
                )
            )
        return positions

    def get_account(self) -> AccountSummary:
        """Equity (available plus held, at USD prices) and cash (available quote funds only).

        Held quote funds are excluded from cash so the risk gate never sizes against money it
        cannot spend. A holding with no USD product is logged and left out of equity.
        """
        cash = Decimal(0)
        equity = Decimal(0)
        for account in self._accounts():
            currency = getattr(account, "currency", "")
            available = _amount(getattr(account, "available_balance", None))
            total = available + _amount(getattr(account, "hold", None))
            if total <= 0:
                continue
            if currency in QUOTE_CURRENCIES:
                cash += available
                equity += total
                continue
            try:
                product = self._client.get_product(f"{currency}-USD")
                equity += total * Decimal(str(product.price))
            except Exception:
                log.warning(
                    "holding_unpriceable_excluded_from_equity",
                    broker="coinbase",
                    currency=currency,
                    balance=str(total),
                )
        return AccountSummary(equity=equity, cash=cash)

    def _accounts(self) -> list:
        """Every account balance, across pages."""
        accounts: list = []
        cursor = None
        try:
            while True:
                kwargs = {"limit": 250}
                if cursor:
                    kwargs["cursor"] = cursor
                page = self._client.get_accounts(**kwargs)
                accounts.extend(getattr(page, "accounts", None) or [])
                if not getattr(page, "has_next", False):
                    return accounts
                cursor = getattr(page, "cursor", None)
        except Exception as e:
            raise BrokerError(f"Coinbase get_accounts failed: {e}") from e

    def _product(self, symbol: str) -> tuple[Decimal, Decimal, Decimal | None]:
        """A product's (base_increment, quote_increment, base_min_size or None), cached."""
        if symbol not in self._products:
            try:
                product = self._client.get_product(symbol)
                base_min = getattr(product, "base_min_size", None)
                self._products[symbol] = (
                    Decimal(str(product.base_increment)),
                    Decimal(str(product.quote_increment)),
                    Decimal(str(base_min)) if base_min is not None else None,
                )
            except Exception as e:
                raise BrokerError(f"Coinbase get_product failed for {symbol}: {e}") from e
        return self._products[symbol]


def _order_id_from(response, symbol: str) -> str:
    """The order id from a create-order response; BrokerError if rejected or missing."""
    if not getattr(response, "success", False):
        error = getattr(response, "error_response", None)
        raise BrokerError(f"Coinbase rejected order for {symbol}: {error!r}")
    success = getattr(response, "success_response", None) or {}
    order_id = success.get("order_id")
    if not order_id:
        raise BrokerError(f"Coinbase order for {symbol} accepted but no order_id returned")
    return str(order_id)


def _parse_ts(value: str | datetime) -> datetime:
    """Parse a venue RFC3339 timestamp ("...Z") to an aware datetime."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _amount(balance) -> Decimal:
    """An SDK Amount dict's value as a Decimal, 0 when absent."""
    if not balance:
        return Decimal(0)
    value = balance.get("value")
    return Decimal(str(value)) if value is not None else Decimal(0)
