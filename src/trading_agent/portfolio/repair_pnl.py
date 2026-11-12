"""Recompute realized P&L booked against lots sold at the ledger start.

    python -m trading_agent.portfolio.repair_pnl           # dry run: list what would change
    python -m trading_agent.portfolio.repair_pnl --write   # write those changes

Every paper closing fill since the ledger start is re-walked against the fills since
that start only, the rule the executor now books with. Each change is logged with
its old value before it is written, so a write can be undone from the log.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from trading_agent.config import load_settings
from trading_agent.graph.nodes.trade_executor import realized_pnl
from trading_agent.log import configure_logging, get_logger
from trading_agent.storage.db import make_engine, make_session_factory
from trading_agent.storage.models import Trade
from trading_agent.storage.repos import CashFlowRepository

log = get_logger(__name__)

# The column keeps 10 decimals: closer than this is the same booked value.
SAME = Decimal("1e-8")


@dataclass(frozen=True)
class Correction:
    """One closing fill whose booked realized P&L differs from the recomputed one."""

    trade_id: int
    symbol: str
    filled_at: datetime
    booked: Decimal | None
    fixed: Decimal | None


def corrections(session: Session, trading_mode: str) -> list[Correction]:
    """The closing fills since the ledger start whose realized P&L would change."""
    start = CashFlowRepository(session).start(trading_mode)
    if start is None:
        return []
    fills = (
        session.query(Trade)
        .filter(Trade.trading_mode == trading_mode, Trade.filled_at >= start)
        .order_by(Trade.filled_at.asc(), Trade.id.asc())
        .all()
    )
    out: list[Correction] = []
    seen: dict[str, list[Trade]] = {}
    for fill in fills:
        prior = seen.setdefault(fill.symbol, [])
        if fill.side == "sell":
            fixed = realized_pnl(prior, fill.fill_price, fill.quantity, fill.fees or Decimal(0))
            booked = fill.realized_pnl
            if (booked is None) != (fixed is None) or (
                booked is not None and abs(booked - fixed) > SAME
            ):
                out.append(Correction(fill.id, fill.symbol, fill.filled_at, booked, fixed))
        prior.append(fill)
    return out


def write(session: Session, changes: list[Correction]) -> None:
    """Write each correction, logging its old value first (the caller commits)."""
    for c in changes:
        log.info(
            "realized_pnl_corrected",
            trade_id=c.trade_id,
            symbol=c.symbol,
            old=None if c.booked is None else str(c.booked),
            new=None if c.fixed is None else str(c.fixed),
        )
        session.get(Trade, c.trade_id).realized_pnl = c.fixed


def main(argv: list[str] | None = None) -> None:
    """CLI: list the corrections, and write them with --write (paper mode only)."""
    parser = argparse.ArgumentParser(prog="trading_agent.portfolio.repair_pnl")
    parser.add_argument("--write", action="store_true", help="write the listed corrections")
    args = parser.parse_args(argv)
    configure_logging()
    settings = load_settings()
    if not settings.is_paper:
        raise SystemExit("the paper ledger exists in paper mode only")
    factory = make_session_factory(make_engine(settings.database_url))
    with factory() as session:
        changes = corrections(session, settings.trading_mode)
        for c in changes:
            when = f"{c.filled_at:%Y-%m-%d %H:%M}"
            print(f"{c.trade_id:>8} {c.symbol:<8} {when} {c.booked} -> {c.fixed}")
        if not changes:
            print("nothing to correct")
        elif args.write:
            write(session, changes)
            session.commit()
            print(f"wrote {len(changes)} correction(s)")
        else:
            print("dry run: nothing written (pass --write)")


if __name__ == "__main__":
    main()
