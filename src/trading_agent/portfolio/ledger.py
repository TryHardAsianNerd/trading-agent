"""The paper ledger: the plan's deposits simulated at real scale, in paper mode only (D-040).

`python -m trading_agent.portfolio.ledger start [--close-positions]`, with the agent stopped,
records a baseline row per venue cancelling its play money, then the first deposit. What the
app owns at a venue is the venue's figures plus its ledger rows.
"""

import argparse
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy.exc import IntegrityError

from trading_agent.brokers.base import Broker, BrokerError
from trading_agent.config import load_settings
from trading_agent.graph.state import ApprovedOrder
from trading_agent.log import configure_logging
from trading_agent.storage.db import make_engine, make_session_factory
from trading_agent.storage.models import CashFlow
from trading_agent.storage.repos import CashFlowRepository

# The backtest simulates the same schedule.
FIRST_DEPOSIT = Decimal(1000)
DEPOSIT = Decimal(500)
DEPOSIT_EVERY = timedelta(days=14)
# The crypto cap: that share of every deposit lands at the crypto venue.
CRYPTO_SHARE = Decimal("0.10")
# The core stocks' share of every deposit. Each row records its split when booked, so
# a change here applies to later deposits only.
CORE_STOCK_SHARE = Decimal("0.60")
CLOSE_TIMEOUT = timedelta(minutes=5)
CLOSE_POLL_SECONDS = 5.0


def split(amount: Decimal, venues: set[str]) -> dict[str, Decimal]:
    """Divide a deposit: CRYPTO_SHARE to crypto and the rest to stocks, or all to a lone venue."""
    if len(venues) == 1:
        return {next(iter(venues)): amount}
    crypto = amount * CRYPTO_SHARE
    return {"crypto": crypto, "stock": amount - crypto}


def core_split(amount: Decimal, venues: set[str]) -> dict[str, Decimal]:
    """The core's part of each venue's share: all of crypto's, CORE_STOCK_SHARE at stocks."""
    return {
        venue: part if venue == "crypto" else CORE_STOCK_SHARE * amount
        for venue, part in split(amount, venues).items()
    }


def deposit_rows(
    amount: Decimal,
    venues: set[str],
    equity_before: Decimal,
    ts: datetime,
    scheduled_for: date,
    trading_mode: str,
) -> list[CashFlow]:
    """Unsaved deposit rows for one deposit, one per venue.

    Each row's equity_before includes the rows before it, so their growth factors multiply
    to the whole deposit's.
    """
    rows = []
    core = core_split(amount, venues)
    for venue, part in sorted(split(amount, venues).items()):
        rows.append(
            CashFlow(
                ts=ts,
                trading_mode=trading_mode,
                asset_class=venue,
                kind="deposit",
                amount=part,
                core_amount=core[venue],
                equity_before=equity_before,
                scheduled_for=scheduled_for,
            )
        )
        equity_before += part
    return rows


def due_dates(flows: list[CashFlow], today: date) -> list[date]:
    """Schedule dates after the baseline, up to today, that have no deposit yet, oldest first."""
    start = flows[0].ts.date()
    done = {f.scheduled_for for f in flows if f.kind == "deposit"}
    due, day = [], start + DEPOSIT_EVERY
    while day <= today:
        if day not in done:
            due.append(day)
        day += DEPOSIT_EVERY
    return due


def offsets(flows: list[CashFlow]) -> dict[str, Decimal]:
    """What the ledger adds to each venue's own cash and equity."""
    out: dict[str, Decimal] = {}
    for f in flows:
        out[f.asset_class] = out.get(f.asset_class, Decimal(0)) + f.amount
    return out


def growth_after(flows: list[CashFlow], after: datetime) -> Decimal:
    """The factor by which deposits booked after `after` scaled the account (1 when none).

    A breaker anchor times this factor folds the later deposits in, so a deposit reads as
    neither a gain nor a smaller loss.
    """
    factor = Decimal(1)
    for f in flows:
        if f.kind == "deposit" and f.ts > after and f.equity_before > 0:
            factor *= (f.equity_before + f.amount) / f.equity_before
    return factor


def close_positions(
    brokers: list[Broker],
    now: datetime,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[str]:
    """Cancel every open order, sell every position and wait until flat; return "venue:symbol"s.

    A refused sell is retried until CLOSE_TIMEOUT, because Alpaca cancels in the background
    and a stop still pending its cancel holds the shares. ValueError if anything is still held.
    """
    for broker in brokers:
        broker.cancel_all_orders()
    unsold = [(b, p) for b in brokers for p in b.get_positions()]
    sold: list[str] = []
    refused: dict[str, str] = {}
    deadline = clock() + CLOSE_TIMEOUT
    while True:
        retry = []
        for broker, position in unsold:
            name = f"{broker.name}:{position.symbol}"
            order = ApprovedOrder(
                symbol=position.symbol,
                asset_class=broker.asset_class,
                side="sell",
                quantity=position.quantity,
                stop_loss_price=None,
                source_proposal=None,
            )
            try:
                broker.submit_order(order, f"cutover-{now:%Y%m%d}:{position.symbol}")
            except BrokerError as e:
                refused[name] = str(e)
                retry.append((broker, position))
                continue
            refused.pop(name, None)
            sold.append(name)
        unsold = retry
        held = [f"{b.name}:{p.symbol}" for b in brokers for p in b.get_positions()]
        if not held:
            return sold
        if clock() >= deadline:
            reasons = "".join(f"; {n} refused: {r}" for n, r in refused.items())
            raise ValueError(f"still holding {', '.join(held)}: is the NYSE session open?{reasons}")
        sleep(CLOSE_POLL_SECONDS)


def start_ledger(brokers: list[Broker], now: datetime) -> list[CashFlow]:
    """A baseline row per venue cancelling its cash, then the FIRST_DEPOSIT rows.

    The ledger is read from the latest baseline on, so this also restarts it. ValueError if
    a venue holds positions or none is configured.
    """
    if not brokers:
        raise ValueError("no venues configured")
    rows = []
    for broker in brokers:
        held = broker.get_positions()
        if held:
            names = ", ".join(p.symbol for p in held)
            raise ValueError(f"{broker.name} holds {names}: close them before starting")
        rows.append(
            CashFlow(
                ts=now,
                trading_mode="paper",
                asset_class=broker.asset_class,
                kind="baseline",
                amount=-broker.get_account().cash,
                core_amount=Decimal(0),
                equity_before=Decimal(0),
            )
        )
    venues = {b.asset_class for b in brokers}
    first = deposit_rows(FIRST_DEPOSIT, venues, Decimal(0), now, now.date(), "paper")
    for row in first:
        row.equity_before = Decimal(0)
    return rows + first


def main(argv: list[str] | None = None) -> None:
    """CLI: start the paper ledger, optionally selling everything first."""
    # Stop the agent first: a cycle that snapshots the old $110k account after the
    # start would become the new ledger's peak, and the kill switch would trip.
    parser = argparse.ArgumentParser(
        prog="trading_agent.portfolio.ledger",
        description="Start the paper ledger. Stop the agent and cancel open orders first.",
    )
    parser.add_argument("command", choices=["start"])
    parser.add_argument(
        "--close-positions",
        action="store_true",
        help="cancel every open order and sell every position first (NYSE session open)",
    )
    args = parser.parse_args(argv)
    configure_logging()
    settings = load_settings()
    if not settings.is_paper:
        raise SystemExit("the ledger simulates deposits in paper mode only")
    # Local import: graph.deps imports the tracker, which imports this module.
    from trading_agent.graph.deps import make_deps

    factory = make_session_factory(make_engine(settings.database_url))
    brokers = make_deps(settings, factory).brokers
    try:
        if args.close_positions:
            for name in close_positions(brokers, datetime.now(UTC)):
                print(f"sold     {name}")
        rows = start_ledger(brokers, datetime.now(UTC))
    except (ValueError, BrokerError) as e:
        raise SystemExit(str(e)) from e
    with factory() as session:
        CashFlowRepository(session).add(rows)
        try:
            session.commit()
        except IntegrityError as e:
            raise SystemExit("the previous ledger booked a deposit today: start tomorrow") from e
    for row in rows:
        print(f"{row.kind:<9}{row.asset_class:<8}{row.amount:>14,.2f}")


if __name__ == "__main__":
    main()
