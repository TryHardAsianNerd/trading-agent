"""Broker interface: the minimal surface the executor needs.

Each implementation trades one asset_class and refuses orders for any other. Fills are
reconciled by point lookup on the venue order id, because both real venues filter order
listings by submission time and a GTC stop can fill weeks after it was placed.
"""

from datetime import datetime
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel

from trading_agent.graph.state import ApprovedOrder


class BrokerError(RuntimeError):
    """A venue call failed (transport, rejection or malformed response); may be transient."""


class WrongVenueError(BrokerError):
    """An order's asset_class does not match the venue this broker trades; never retry."""


class UnfillableQuantityError(BrokerError):
    """The order quantity rounds to zero under the venue's lot constraints; never retry."""


class UnknownOrderError(BrokerError):
    """The venue has no record of a booked order id; the executor retires the row as lost."""


class UnprotectedPositionError(BrokerError):
    """An entry filled but its protecting stop could not be placed.

    Never retried blindly: resubmitting the entry doubles the position. The caller must
    alert a human and still book the fill, which is why the entry's venue id travels here.
    """

    def __init__(self, message: str, broker_order_id: str | None = None) -> None:
        """Build the error with the filled entry's venue id, when known."""
        super().__init__(message)
        self.broker_order_id = broker_order_id


class SubmittedOrder(BaseModel):
    """Venue acknowledgement of a submitted order.

    submitted_quantity is what was sent after lot rounding. filled_quantity is set only
    when the venue reports fills synchronously (Coinbase IOC entries). protective_order_id
    is the stop placed with a buy, booked so its venue-side fill is reconciled.
    """

    broker_order_id: str
    status: str
    submitted_quantity: Decimal
    filled_quantity: Decimal | None = None
    client_order_id: str | None = None
    protective_order_id: str | None = None


class VenueOrder(BaseModel):
    """One order's current venue-side state; terminal means it will never fill further."""

    status: str
    terminal: bool
    filled_quantity: Decimal = Decimal(0)
    fill_price: Decimal | None = None
    filled_at: datetime | None = None
    fees: Decimal = Decimal(0)


class BrokerPosition(BaseModel):
    """A held position as the venue reports it.

    avg_entry_price is None when the venue reports balances rather than lots (Coinbase);
    the portfolio tracker derives entries from trade history then.
    """

    symbol: str
    asset_class: str
    quantity: Decimal
    avg_entry_price: Decimal | None = None


class AccountSummary(BaseModel):
    """Account equity and cash as reported by the venue."""

    equity: Decimal
    cash: Decimal


class Broker(Protocol):
    """What the executor may do with a venue; name is recorded on order and trade rows."""

    name: str
    asset_class: str

    def submit_order(
        self, order: ApprovedOrder, client_order_id: str | None = None
    ) -> SubmittedOrder:
        """Submit a gate-approved order; a retry with the same client_order_id must not resubmit.

        Sells are full closes and must succeed even when a working stop reserves the shares.
        Raises WrongVenueError or UnfillableQuantityError (permanent) or BrokerError.
        """
        ...

    def get_order(self, broker_order_id: str) -> VenueOrder:
        """Look up one order's current state; UnknownOrderError if the venue has no record."""
        ...

    def place_stop(
        self, symbol: str, quantity: Decimal, stop_price: Decimal, client_order_id: str
    ) -> str:
        """Place a GTC sell stop on a held position and return its venue id.

        Re-protects a position that lost its stop (e.g. the kill switch cancelled it). A replay
        of the same client_order_id returns the original stop's id. Raises
        UnfillableQuantityError if the quantity rounds to zero at the venue.
        """
        ...

    def cancel_all_orders(self) -> int:
        """Cancel every open order at the venue and return how many it cancelled."""
        ...

    def get_positions(self) -> list[BrokerPosition]:
        """Every open position, one per held symbol."""
        ...

    def get_account(self) -> AccountSummary:
        """Account equity and cash."""
        ...
