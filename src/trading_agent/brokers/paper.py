"""Simulated venue for crypto paper trading; the Coinbase sandbox cannot produce real fills.

Same Broker contract as the real venues. All venue state lives in the paper_* tables so it
survives restarts. Fills price at quote * (1 +/- FILL_COST_PCT), folding slippage and the
Coinbase taker fee into the price. Stops settle lazily, at the observed quote, whenever
get_positions or get_account runs. One process owns an account; a lock serializes the
cycle and Discord bot threads, since two concurrent settles could credit a stop twice.
"""

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.orm import Session, sessionmaker

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
from trading_agent.config import Settings
from trading_agent.graph.state import ApprovedOrder
from trading_agent.log import get_logger
from trading_agent.storage.models import PaperBalance, PaperOrder, PaperPosition

log = get_logger(__name__)

SLIPPAGE_PCT = Decimal("0.002")  # 0.2%: middle of RISK_RULES' 0.1-0.5% band
# Coinbase Advanced entry-tier (<$10k/30 days) taker fee per side; re-check the schedule.
TAKER_FEE_PCT = Decimal("0.006")
FILL_COST_PCT = SLIPPAGE_PCT + TAKER_FEE_PCT


class PaperOnlyBrokerError(RuntimeError):
    """Raised when a PaperBroker is constructed outside paper mode."""

    def __init__(self) -> None:
        """Build the error with its fixed message."""
        super().__init__(
            "PaperBroker is a simulated venue for TRADING_MODE=paper only - "
            "live mode must route to the real venue (D-013)."
        )


class PaperBroker:
    """Simulated venue implementing the Broker protocol against DB state."""

    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        quote_fn: Callable[[str], Decimal],
        asset_class: str = "crypto",
        account: str = "crypto",
    ) -> None:
        """Build the venue; PaperOnlyBrokerError outside paper mode.

        quote_fn may raise: a submit surfaces that as BrokerError, stop settlement skips the
        symbol. account scopes this venue's rows.
        """
        if not settings.is_paper:
            raise PaperOnlyBrokerError()
        self.name = "paper"
        self.asset_class = asset_class
        self._lock = threading.Lock()
        self._session_factory = session_factory
        self._quote = quote_fn
        self._account = account
        self._starting_cash = Decimal(settings.paper_starting_cash)

    def submit_order(
        self, order: ApprovedOrder, client_order_id: str | None = None
    ) -> SubmittedOrder:
        """Fill a gate-approved order at the quote; a reused client_order_id returns the original.

        Buys record their GTC stop atomically; sells cancel the symbol's stops and close it fully.
        """
        if order.asset_class != self.asset_class:
            raise WrongVenueError(
                f"PaperBroker({self.asset_class}) cannot trade {order.asset_class} "
                f"order for {order.symbol}"
            )
        with self._lock, self._session_factory() as session:
            self._ensure_account(session)
            if client_order_id:
                existing = (
                    session.query(PaperOrder)
                    .filter(
                        PaperOrder.account == self._account,
                        PaperOrder.client_order_id == client_order_id,
                    )
                    .first()
                )
                if existing is not None:
                    log.warning(
                        "paper_duplicate_client_order_id",
                        symbol=order.symbol,
                        client_order_id=client_order_id,
                    )
                    return self._submitted(existing)
            row = (
                self._fill_buy(session, order, client_order_id)
                if order.side == "buy"
                else self._fill_sell(session, order, client_order_id)
            )
            session.commit()
            log.info(
                "paper_order_filled",
                symbol=order.symbol,
                side=order.side,
                quantity=str(row.quantity),
                fill_price=str(row.fill_price),
            )
            return self._submitted(row)

    def cancel_all_orders(self) -> int:
        """Cancel every open stop and return how many."""
        with self._lock, self._session_factory() as session:
            open_orders = self._open_stops(session)
            for stop in open_orders:
                stop.status = "cancelled"
            session.commit()
            return len(open_orders)

    def get_order(self, broker_order_id: str) -> VenueOrder:
        """Report one order's state; stops are settled by get_positions/get_account, not here."""
        try:
            row_id = int(broker_order_id)
        except ValueError as e:
            raise UnknownOrderError(f"malformed paper order id {broker_order_id!r}") from e
        with self._lock, self._session_factory() as session:
            row = session.get(PaperOrder, row_id)
            if row is None or row.account != self._account:
                raise UnknownOrderError(
                    f"unknown paper order {broker_order_id} for {self._account}"
                )
            return VenueOrder(
                status=row.status,
                terminal=row.status != "open",
                filled_quantity=row.filled_quantity or Decimal(0),
                fill_price=row.fill_price,
                filled_at=row.filled_at,
            )

    def get_positions(self) -> list[BrokerPosition]:
        """Settle crossed stops, then return the held positions."""
        with self._lock, self._session_factory() as session:
            self._ensure_account(session)
            self._settle_stops(session)
            session.commit()  # settle results persist even if the read below fails
            rows = (
                session.query(PaperPosition)
                .filter(PaperPosition.account == self._account, PaperPosition.quantity > 0)
                .all()
            )
            return [
                BrokerPosition(
                    symbol=p.symbol,
                    asset_class=self.asset_class,
                    quantity=p.quantity,
                    avg_entry_price=p.avg_entry_price,
                )
                for p in rows
            ]

    def get_account(self) -> AccountSummary:
        """Settle crossed stops, then return cash and equity marked at current quotes.

        A position whose quote fails is marked at its average entry, with a warning.
        """
        with self._lock, self._session_factory() as session:
            balance = self._ensure_account(session)
            quotes = self._settle_stops(session)
            session.commit()
            equity = balance.cash
            positions = (
                session.query(PaperPosition)
                .filter(PaperPosition.account == self._account, PaperPosition.quantity > 0)
                .all()
            )
            for position in positions:
                # Reuse the settle pass quotes: one per symbol per observation (rate limits).
                price = quotes.get(position.symbol)
                if price is None:
                    try:
                        price = self._quote(position.symbol)
                    except Exception:
                        log.warning("paper_mark_fallback_to_entry", symbol=position.symbol)
                        price = position.avg_entry_price
                equity += position.quantity * price
            return AccountSummary(equity=equity, cash=balance.cash)

    def place_stop(
        self, symbol: str, quantity: Decimal, stop_price: Decimal, client_order_id: str
    ) -> str:
        """Record a GTC sell stop on a held position; return its id, or the original on a replay."""
        if quantity <= 0:
            raise UnfillableQuantityError(f"stop quantity {quantity} for {symbol}")
        with self._lock, self._session_factory() as session:
            existing = (
                session.query(PaperOrder)
                .filter(
                    PaperOrder.account == self._account,
                    PaperOrder.client_order_id == client_order_id,
                )
                .first()
            )
            if existing is not None:
                return str(existing.id)
            row = PaperOrder(
                account=self._account,
                created_at=datetime.now(UTC),
                symbol=symbol,
                side="sell",
                order_type="stop",
                quantity=quantity,
                stop_price=stop_price,
                status="open",
                client_order_id=client_order_id,
            )
            session.add(row)
            session.commit()
            log.info(
                "paper_stop_placed",
                symbol=symbol,
                quantity=str(quantity),
                stop_price=str(stop_price),
            )
            return str(row.id)

    def _fill_buy(
        self, session: Session, order: ApprovedOrder, client_order_id: str | None
    ) -> PaperOrder:
        """Fill a buy at quote * (1 + FILL_COST_PCT), updating cash, the position and its stop."""
        if order.quantity <= 0:
            raise UnfillableQuantityError(f"quantity {order.quantity} for {order.symbol}")
        fill_price = self._quoted(order.symbol) * (1 + FILL_COST_PCT)
        cost = fill_price * order.quantity
        balance = self._ensure_account(session)
        if cost > balance.cash:
            raise BrokerError(
                f"insufficient simulated cash for {order.symbol}: need {cost}, have {balance.cash}"
            )
        balance.cash -= cost
        position = session.get(PaperPosition, (self._account, order.symbol))
        if position is None:  # flat positions are always deleted, never zero rows
            session.add(
                PaperPosition(
                    account=self._account,
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_entry_price=fill_price,
                )
            )
        else:
            total = position.avg_entry_price * position.quantity + cost
            position.quantity += order.quantity
            position.avg_entry_price = total / position.quantity
        row = self._order_row(order, "market", order.quantity, fill_price, client_order_id)
        session.add(row)
        session.flush()  # row.id is the stop's parent link
        if order.stop_loss_price is not None:
            session.add(
                PaperOrder(
                    account=self._account,
                    created_at=datetime.now(UTC),
                    symbol=order.symbol,
                    side="sell",
                    order_type="stop",
                    quantity=order.quantity,
                    stop_price=order.stop_loss_price,
                    status="open",
                    parent_order_id=row.id,
                )
            )
            session.flush()
        return row

    def _fill_sell(
        self, session: Session, order: ApprovedOrder, client_order_id: str | None
    ) -> PaperOrder:
        """Cancel the symbol's stops and sell the whole position, whatever order.quantity says."""
        position = session.get(PaperPosition, (self._account, order.symbol))
        if position is None or position.quantity <= 0:
            raise BrokerError(f"no simulated position to sell in {order.symbol}")
        if order.quantity != position.quantity:
            log.warning(
                "paper_sell_quantity_mismatch_closing_full",
                symbol=order.symbol,
                order_qty=str(order.quantity),
                held=str(position.quantity),
            )
        for stop in self._open_stops(session, symbol=order.symbol):
            stop.status = "cancelled"
        fill_price = self._quoted(order.symbol) * (1 - FILL_COST_PCT)
        quantity = position.quantity
        self._ensure_account(session).cash += fill_price * quantity
        session.delete(position)
        row = self._order_row(order, "market", quantity, fill_price, client_order_id)
        session.add(row)
        session.flush()
        return row

    def _settle_stops(self, session: Session) -> dict[str, Decimal]:
        """Fill open stops whose quote crossed the trigger; return the quotes used, by symbol.

        Fills at the observed quote, so a gap through the stop fills worse. Each symbol is quoted
        once; a failed quote leaves its stops open. The stop row keeps its requested quantity and
        records the fill, capped at the remaining position, in filled_quantity. Never raises.
        """
        quotes: dict[str, Decimal] = {}
        stops = self._open_stops(session)
        for symbol in {s.symbol for s in stops}:
            try:
                quotes[symbol] = self._quote(symbol)
            except Exception:
                log.warning("paper_stop_check_skipped", symbol=symbol, reason="quote failed")
        for stop in stops:
            quote = quotes.get(stop.symbol)
            if quote is None or quote > stop.stop_price:
                continue
            position = session.get(PaperPosition, (self._account, stop.symbol))
            # Quantity guard too: an earlier stop in this pass may have consumed the position.
            if position is None or position.quantity <= 0:
                stop.status = "cancelled"
                continue
            quantity = min(stop.quantity, position.quantity)
            fill_price = quote * (1 - FILL_COST_PCT)
            self._ensure_account(session).cash += fill_price * quantity
            position.quantity -= quantity
            if position.quantity <= 0:
                session.delete(position)
            stop.status = "filled"
            stop.fill_price = fill_price
            stop.filled_quantity = quantity
            stop.filled_at = datetime.now(UTC)
            log.warning(
                "paper_stop_triggered",
                symbol=stop.symbol,
                stop_price=str(stop.stop_price),
                fill_price=str(fill_price),
                quantity=str(quantity),
            )
        return quotes

    def _ensure_account(self, session: Session) -> PaperBalance:
        """This venue's balance row, created with the starting cash the first time."""
        balance = session.get(PaperBalance, self._account)
        if balance is None:
            balance = PaperBalance(account=self._account, cash=self._starting_cash)
            session.add(balance)
            session.flush()
            log.info("paper_account_created", account=self._account, cash=str(balance.cash))
        return balance

    def _open_stops(self, session: Session, symbol: str | None = None) -> list[PaperOrder]:
        """Open stop orders for this account, optionally for one symbol."""
        query = session.query(PaperOrder).filter(
            PaperOrder.account == self._account,
            PaperOrder.order_type == "stop",
            PaperOrder.status == "open",
        )
        if symbol is not None:
            query = query.filter(PaperOrder.symbol == symbol)
        return query.all()

    def _quoted(self, symbol: str) -> Decimal:
        """The current quote; BrokerError if the quote fails or is not positive."""
        try:
            price = self._quote(symbol)
        except Exception as e:
            raise BrokerError(f"paper quote failed for {symbol}: {type(e).__name__}") from e
        if price <= 0:
            raise BrokerError(f"paper quote non-positive for {symbol}: {price}")
        return price

    def _order_row(
        self,
        order: ApprovedOrder,
        order_type: str,
        quantity: Decimal,
        fill_price: Decimal,
        client_order_id: str | None,
    ) -> PaperOrder:
        """Build an unsaved filled order row."""
        now = datetime.now(UTC)
        return PaperOrder(
            account=self._account,
            created_at=now,
            symbol=order.symbol,
            side=order.side,
            order_type=order_type,
            quantity=quantity,
            fill_price=fill_price,
            filled_quantity=quantity,
            filled_at=now,
            status="filled",
            client_order_id=client_order_id,
        )

    def _submitted(self, row: PaperOrder) -> SubmittedOrder:
        """SubmittedOrder for a fill row; the stop comes by parent link so a replay matches."""
        session = Session.object_session(row)
        stop = (
            session.query(PaperOrder.id)
            .filter(PaperOrder.parent_order_id == row.id, PaperOrder.order_type == "stop")
            .first()
        )
        return SubmittedOrder(
            broker_order_id=str(row.id),
            status=row.status,
            submitted_quantity=row.quantity,
            filled_quantity=row.filled_quantity if row.status == "filled" else None,
            client_order_id=row.client_order_id,
            protective_order_id=str(stop[0]) if stop else None,
        )
