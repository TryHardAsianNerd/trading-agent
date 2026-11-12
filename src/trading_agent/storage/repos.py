"""Repositories: thin, typed persistence operations.

PostgreSQL only (the upserts use INSERT .. ON CONFLICT). Repositories never commit; the
caller owns the transaction.
"""

from collections.abc import Collection
from datetime import datetime
from decimal import Decimal

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Query, Session

from trading_agent.data.models import Candle
from trading_agent.storage.models import (
    CashFlow,
    Decision,
    JobRun,
    NewsSentiment,
    Ohlcv,
    Order,
    PortfolioSnapshot,
    Trade,
)

# orders.status values never polled again ("lost": venue has no record or fill unbookable).
TERMINAL_ORDER_STATUSES = ("filled", "cancelled", "lost")
# Wins a day two providers both stored: it serves both asset classes when its keys are set.
PREFERRED_CANDLE_SOURCE = "alpaca"


class OhlcvRepository:
    def __init__(self, session: Session) -> None:
        """Bind the repository to a session."""
        self._session = session

    def upsert_candles(self, candles: list[Candle]) -> int:
        """Upsert candles so provider corrections overwrite stored rows; return the count sent."""
        if not candles:
            return 0
        rows = [
            {
                "symbol": c.symbol,
                "ts": c.ts,
                "source": c.source,
                "resolution": c.resolution,
                "asset_class": c.asset_class,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        stmt = insert(Ohlcv).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "ts", "source", "resolution"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )
        self._session.execute(stmt)
        return len(rows)

    def recent_daily_candles(self, symbol: str, limit: int = 60) -> list[Candle]:
        """A symbol's latest `limit` daily candles, oldest first, one per day.

        Rows with source "test" are excluded so integration-test data never reaches production
        reads. Where two providers stored the same day, PREFERRED_CANDLE_SOURCE wins, then the
        source name, so the pick is stable and the 200-day line counts each day once.
        """
        rows = (
            self._session.query(Ohlcv)
            .filter(Ohlcv.symbol == symbol, Ohlcv.resolution == "1d", Ohlcv.source != "test")
            .distinct(Ohlcv.ts)  # PostgreSQL DISTINCT ON: the first row per ts in this order
            .order_by(
                Ohlcv.ts.desc(), (Ohlcv.source == PREFERRED_CANDLE_SOURCE).desc(), Ohlcv.source
            )
            .limit(limit)
            .all()
        )
        return [
            Candle(
                symbol=r.symbol,
                asset_class=r.asset_class,
                ts=r.ts,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                source=r.source,
                resolution=r.resolution,
            )
            for r in reversed(rows)
        ]


class DecisionRepository:
    def __init__(self, session: Session) -> None:
        """Bind the repository to a session."""
        self._session = session

    def log(
        self,
        *,
        cycle_id: str,
        ts: datetime,
        node: str,
        detail: dict,
        symbol: str | None = None,
        action: str | None = None,
        note: str | None = None,
    ) -> None:
        """Append one row to the decisions audit log; detail must be JSON-serializable."""
        self._session.add(
            Decision(
                cycle_id=cycle_id,
                ts=ts,
                node=node,
                symbol=symbol,
                action=action,
                detail=detail,
                note=note,
            )
        )


class OrderRepository:
    """Our book of venue orders: what the executor submitted and still polls."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session."""
        self._session = session

    def ensure(
        self,
        *,
        broker: str,
        broker_order_id: str,
        created_at: datetime,
        symbol: str,
        asset_class: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        stop_loss_price: Decimal | None,
        status: str,
        trading_mode: str,
    ) -> Order:
        """Return the orders row for a venue order id, creating and flushing it if absent.

        A resumed cycle re-books the same venue ids, so this must not duplicate rows; the unique
        (broker, broker_order_id) index backs that at the database.
        """
        row = (
            self._session.query(Order)
            .filter(Order.broker == broker, Order.broker_order_id == broker_order_id)
            .first()
        )
        if row is None:
            row = Order(
                created_at=created_at,
                symbol=symbol,
                asset_class=asset_class,
                side=side,
                quantity=quantity,
                order_type=order_type,
                stop_loss_price=stop_loss_price,
                broker=broker,
                broker_order_id=broker_order_id,
                status=status,
                trading_mode=trading_mode,
            )
            self._session.add(row)
            self._session.flush()
        return row

    def open_orders(self, broker: str, trading_mode: str) -> list[Order]:
        """One venue's non-terminal orders, oldest first so entries precede their stops."""
        return (
            self._session.query(Order)
            .filter(
                Order.broker == broker,
                Order.trading_mode == trading_mode,
                Order.status.notin_(TERMINAL_ORDER_STATUSES),
            )
            .order_by(Order.created_at.asc(), Order.id.asc())
            .all()
        )

    def entry_stop(self, symbol: str, trading_mode: str) -> Decimal | None:
        """The app-held stop for a position: the one on its latest market buy since the last sale.

        A lost buy still counts, since its shares exist. None when that buy carried no stop (a core
        buy) or no buy is booked since the last sale.
        """
        mine = (Order.symbol == symbol, Order.trading_mode == trading_mode)
        last_sale = (
            self._session.query(func.max(Order.created_at))
            .filter(*mine, Order.side == "sell", Order.status == "filled")
            .scalar()
        )
        query = self._session.query(Order).filter(
            *mine, Order.side == "buy", Order.order_type == "market", Order.status != "cancelled"
        )
        if last_sale is not None:
            query = query.filter(Order.created_at > last_sale)
        row = query.order_by(Order.created_at.desc(), Order.id.desc()).first()
        return row.stop_loss_price if row is not None else None

    def has_fill(self, order_id: int) -> bool:
        """Whether a trades row is already booked against an orders row."""
        return self._session.query(Trade.id).filter(Trade.order_id == order_id).first() is not None


class TradeRepository:
    """Fill history: the source of truth for cost basis and loss streaks."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session."""
        self._session = session

    def record_fill(
        self,
        *,
        filled_at: datetime,
        symbol: str,
        asset_class: str,
        side: str,
        quantity: Decimal,
        fill_price: Decimal,
        broker: str,
        trading_mode: str,
        fees: Decimal = Decimal(0),
        realized_pnl: Decimal | None = None,
        order_id: int | None = None,
    ) -> None:
        """Append one fill to the trades table; realized_pnl is set on closing fills only."""
        self._session.add(
            Trade(
                order_id=order_id,
                filled_at=filled_at,
                symbol=symbol,
                asset_class=asset_class,
                side=side,
                quantity=quantity,
                fill_price=fill_price,
                fees=fees,
                broker=broker,
                trading_mode=trading_mode,
                realized_pnl=realized_pnl,
            )
        )

    def fills_for_symbol(
        self, symbol: str, trading_mode: str, since: datetime | None = None
    ) -> list[Trade]:
        """One symbol's fills in one trading mode, oldest first, from since when given.

        Cost basis passes the paper ledger's start as since: the ledger starts flat
        and never books the sales that emptied the account before it.
        """
        query = self._session.query(Trade).filter(
            Trade.symbol == symbol, Trade.trading_mode == trading_mode
        )
        if since is not None:
            query = query.filter(Trade.filled_at >= since)
        return query.order_by(Trade.filled_at.asc(), Trade.id.asc()).all()

    def venue_fills(self, asset_class: str, trading_mode: str, since: datetime) -> list[Trade]:
        """Every fill at one venue (asset class) since a time, oldest first."""
        return (
            self._session.query(Trade)
            .filter(
                Trade.asset_class == asset_class,
                Trade.trading_mode == trading_mode,
                Trade.filled_at >= since,
            )
            .order_by(Trade.filled_at.asc(), Trade.id.asc())
            .all()
        )

    def trailing_loss_streak(
        self, trading_mode: str, since: datetime, exclude: Collection[str] = ()
    ) -> int:
        """Count consecutive losing closes since `since`, newest first, ignoring `exclude` symbols.

        The window lets the consecutive-losses breaker heal: an old streak with no closes since
        must not halt trading forever.
        """
        closes = (
            self._session.query(Trade.realized_pnl)
            .filter(
                Trade.side == "sell",
                Trade.realized_pnl.isnot(None),
                Trade.trading_mode == trading_mode,
                Trade.filled_at >= since,
                Trade.symbol.notin_(exclude),
            )
            .order_by(Trade.filled_at.desc(), Trade.id.desc())
            .limit(50)
            .all()
        )
        streak = 0
        for (pnl,) in closes:
            if pnl < 0:
                streak += 1
            else:
                break
        return streak


class SnapshotRepository:
    """Portfolio snapshots: history the tracker derives period anchors from."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session."""
        self._session = session

    def record(
        self,
        *,
        ts: datetime,
        equity: Decimal,
        cash: Decimal,
        positions: dict,
        peak_equity: Decimal,
        trading_mode: str,
        swing: tuple[Decimal, Decimal] | None = None,
    ) -> None:
        """Persist one snapshot; swing is (swing equity, swing index) while a paper ledger runs."""
        swing_equity, swing_index = swing or (None, None)
        self._session.merge(  # merge: same-ts re-snapshot updates instead of PK-colliding
            PortfolioSnapshot(
                ts=ts,
                equity=equity,
                cash=cash,
                positions=positions,
                peak_equity=peak_equity,
                trading_mode=trading_mode,
                swing_equity=swing_equity,
                swing_index=swing_index,
            )
        )

    def snapshot_at(
        self,
        boundary: datetime,
        trading_mode: str,
        since: datetime | None = None,
        swing_only: bool = False,
    ) -> PortfolioSnapshot | None:
        """The snapshot anchoring a period boundary for the daily and weekly breakers.

        The latest snapshot at or before the boundary, so losses between the boundary and the next
        observation (overnight, downtime) count. Falls back to the first one after it only when
        nothing precedes it. since drops older snapshots (the paper ledger's start); swing_only
        keeps snapshots that measured the swing sleeve.
        """
        query = self._scoped(trading_mode, since)
        if swing_only:
            query = query.filter(PortfolioSnapshot.swing_index.isnot(None))
        before = (
            query.filter(PortfolioSnapshot.ts <= boundary)
            .order_by(PortfolioSnapshot.ts.desc())
            .first()
        )
        if before:
            return before
        return query.filter(PortfolioSnapshot.ts >= boundary).order_by(PortfolioSnapshot.ts).first()

    def latest_swing(self, trading_mode: str, since: datetime) -> PortfolioSnapshot | None:
        """The newest snapshot that measured the swing sleeve, or None."""
        return (
            self._scoped(trading_mode, since)
            .filter(PortfolioSnapshot.swing_index.isnot(None))
            .order_by(PortfolioSnapshot.ts.desc())
            .first()
        )

    def peak_swing_index(self, trading_mode: str, since: datetime) -> Decimal | None:
        """The highest swing index since the paper ledger started, or None."""
        return (
            self._scoped(trading_mode, since)
            .with_entities(func.max(PortfolioSnapshot.swing_index))
            .scalar()
        )

    def latest(self, trading_mode: str, since: datetime | None = None) -> PortfolioSnapshot | None:
        """The newest snapshot; its peak_equity is the running peak.

        Scoped by mode so a paper peak can never arm the live drawdown kill switch.
        """
        return self._scoped(trading_mode, since).order_by(PortfolioSnapshot.ts.desc()).first()

    def _scoped(self, trading_mode: str, since: datetime | None) -> Query[PortfolioSnapshot]:
        """A query for one mode's snapshots, from `since` on when given."""
        query = self._session.query(PortfolioSnapshot).filter(
            PortfolioSnapshot.trading_mode == trading_mode
        )
        return query if since is None else query.filter(PortfolioSnapshot.ts >= since)


class CashFlowRepository:
    """The paper ledger: its baseline and scheduled deposits."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session."""
        self._session = session

    def ledger(self, trading_mode: str) -> list[CashFlow]:
        """Every row from the latest baseline on, oldest first; [] when no ledger was started."""
        start = self.start(trading_mode)
        if start is None:
            return []
        return (
            self._session.query(CashFlow)
            .filter(CashFlow.trading_mode == trading_mode, CashFlow.ts >= start)
            .order_by(CashFlow.ts, CashFlow.id)
            .all()
        )

    def start(self, trading_mode: str) -> datetime | None:
        """When the running ledger started (its latest baseline), or None without one."""
        return (
            self._session.query(func.max(CashFlow.ts))
            .filter(CashFlow.trading_mode == trading_mode, CashFlow.kind == "baseline")
            .scalar()
        )

    def add(self, rows: list[CashFlow]) -> None:
        """Stage rows; a duplicate scheduled deposit fails the commit with IntegrityError."""
        self._session.add_all(rows)


class NewsSentimentRepository:
    """Scored headlines: the worker writes them, the feature engine summarizes them."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session."""
        self._session = session

    def known_ids(self, article_ids: list[int]) -> set[int]:
        """The subset of these article ids already stored."""
        if not article_ids:
            return set()
        rows = self._session.query(NewsSentiment.article_id).filter(
            NewsSentiment.article_id.in_(article_ids)
        )
        return {r.article_id for r in rows}

    def add(self, rows: list[NewsSentiment]) -> None:
        """Store scored headlines, skipping any id already present."""
        if not rows:
            return
        columns = [c.name for c in NewsSentiment.__table__.columns]
        values = [{c: getattr(r, c) for c in columns} for r in rows]
        self._session.execute(insert(NewsSentiment).values(values).on_conflict_do_nothing())

    def summary(self, symbol: str, since: datetime) -> tuple[float, int]:
        """Mean positive-minus-negative score and headline count since a time; (0.0, 0) if none."""
        mean, count = (
            self._session.query(
                func.avg(NewsSentiment.positive - NewsSentiment.negative), func.count()
            )
            .filter(NewsSentiment.symbols.any(symbol), NewsSentiment.created_at >= since)
            .one()
        )
        return (float(mean) if mean is not None else 0.0, int(count))


class JobRunRepository:
    """Heartbeats of the worker's background jobs."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to a session."""
        self._session = session

    def mark_success(self, name: str, at: datetime) -> None:
        """Record that a job just completed."""
        stmt = insert(JobRun).values(name=name, last_success_at=at)
        self._session.execute(
            stmt.on_conflict_do_update(index_elements=["name"], set_={"last_success_at": at})
        )

    def last_success(self, name: str) -> datetime | None:
        """When a job last completed, or None if it never has."""
        run = self._session.get(JobRun, name)
        return run.last_success_at if run else None
