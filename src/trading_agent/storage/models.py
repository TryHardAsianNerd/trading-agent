"""SQLAlchemy models for the one TimescaleDB database.

`ohlcv` becomes a hypertable in db.init_db(). `decisions` is the audit log: a row per cycle
whether or not it traded.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Date, DateTime, Float, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 30 digits / 10 decimals covers satoshi-scale quantities and any realistic USD price losslessly.
NUMERIC = Numeric(precision=30, scale=10)


class Base(DeclarativeBase):
    pass


class Ohlcv(Base):
    __tablename__ = "ohlcv"

    # Composite PK: hypertables require the partition column (ts) in any unique index.
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    source: Mapped[str] = mapped_column(String(20), primary_key=True)
    resolution: Mapped[str] = mapped_column(String(8), primary_key=True, default="1d")

    asset_class: Mapped[str] = mapped_column(String(10))
    open: Mapped[Decimal] = mapped_column(NUMERIC)
    high: Mapped[Decimal] = mapped_column(NUMERIC)
    low: Mapped[Decimal] = mapped_column(NUMERIC)
    close: Mapped[Decimal] = mapped_column(NUMERIC)
    volume: Mapped[Decimal] = mapped_column(NUMERIC)
    # No extra (symbol, ts) index: the composite PK already leads with symbol, ts.


class Order(Base):
    """Our book of venue orders, one row per venue order id, entries and protective stops alike.

    The executor polls these to reconcile venue-side fills. The unique (broker, broker_order_id)
    index, mirrored in db.init_db, makes booking idempotent on a resumed cycle.
    """

    __tablename__ = "orders"
    __table_args__ = (Index("ux_orders_broker_order_id", "broker", "broker_order_id", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(20))
    asset_class: Mapped[str] = mapped_column(String(10))
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[Decimal] = mapped_column(NUMERIC)
    order_type: Mapped[str] = mapped_column(String(16))
    stop_loss_price: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    broker: Mapped[str] = mapped_column(String(20))
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # "filled"/"cancelled"/"lost" are terminal; anything else is the venue's status, still open.
    status: Mapped[str] = mapped_column(String(32), default="new")
    trading_mode: Mapped[str] = mapped_column(String(8))


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(20))
    asset_class: Mapped[str] = mapped_column(String(10))
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[Decimal] = mapped_column(NUMERIC)
    fill_price: Mapped[Decimal] = mapped_column(NUMERIC)
    fees: Mapped[Decimal] = mapped_column(NUMERIC, default=Decimal(0))
    broker: Mapped[str] = mapped_column(String(20))
    trading_mode: Mapped[str] = mapped_column(String(8))
    # Set on closing fills only (None on buys); feeds the consecutive-losses breaker.
    realized_pnl: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    equity: Mapped[Decimal] = mapped_column(NUMERIC)
    cash: Mapped[Decimal] = mapped_column(NUMERIC)
    positions: Mapped[dict] = mapped_column(JSONB)  # {symbol: {qty, avg_price, ...}}
    peak_equity: Mapped[Decimal] = mapped_column(NUMERIC)
    # Mode-scoped so a paper peak can never arm the live drawdown kill switch.
    trading_mode: Mapped[str] = mapped_column(String(8), default="paper")
    # The swing sleeve while a paper ledger runs; NULL otherwise.
    swing_equity: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    swing_index: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)


class CashFlow(Base):
    """Money moved into the app's accounts, and the paper ledger's start.

    A "baseline" (amount < 0) cancels a paper venue's own starting cash so the app sees only
    the money the plan deposits; a "deposit" (amount > 0) is a scheduled paper deposit. App-side
    cash and equity are the venue's figures plus the rows since the latest baseline.
    equity_before is the whole account's app-side equity just before the row; the tracker
    scales its breaker anchors by (equity_before + amount) / equity_before, so a deposit is
    neither a gain nor a dilution of a loss.
    """

    __tablename__ = "cash_flows"
    # One scheduled deposit per venue and date: a re-run can never deposit twice.
    __table_args__ = (
        Index(
            "ux_cash_flows_schedule",
            "trading_mode",
            "asset_class",
            "kind",
            "scheduled_for",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    trading_mode: Mapped[str] = mapped_column(String(8))
    asset_class: Mapped[str] = mapped_column(String(10))  # the venue: "stock" or "crypto"
    kind: Mapped[str] = mapped_column(String(16))  # "baseline" | "deposit"
    amount: Mapped[Decimal] = mapped_column(NUMERIC)
    # The core's part of the amount, fixed when booked (0 on a baseline); the rest of a
    # stock deposit is the swing sleeve's.
    core_amount: Mapped[Decimal] = mapped_column(NUMERIC)
    equity_before: Mapped[Decimal] = mapped_column(NUMERIC)
    scheduled_for: Mapped[date | None] = mapped_column(Date, nullable=True)  # NULL: baseline


class PaperBalance(Base):
    """Simulated venue cash for one PaperBroker account, separate from our `trades` book."""

    __tablename__ = "paper_balances"

    account: Mapped[str] = mapped_column(String(20), primary_key=True)
    cash: Mapped[Decimal] = mapped_column(NUMERIC)


class PaperPosition(Base):
    """Simulated venue position, with lots so PaperBroker can report an average entry."""

    __tablename__ = "paper_positions"

    account: Mapped[str] = mapped_column(String(20), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    quantity: Mapped[Decimal] = mapped_column(NUMERIC)
    avg_entry_price: Mapped[Decimal] = mapped_column(NUMERIC)


class PaperOrder(Base):
    """Simulated venue order log: filled entries (for idempotency) and GTC stops."""

    __tablename__ = "paper_orders"
    # DB-enforced idempotency (NULL keys distinct); mirror of the index in db.init_db.
    __table_args__ = (
        Index("ux_paper_orders_account_client", "account", "client_order_id", unique=True),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    account: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(20))
    side: Mapped[str] = mapped_column(String(4))
    order_type: Mapped[str] = mapped_column(String(16))  # "market" | "stop"
    quantity: Mapped[Decimal] = mapped_column(NUMERIC)
    stop_price: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    fill_price: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    # Actual filled size (a stop fills at most the remaining position) + time, for get_order.
    filled_quantity: Mapped[Decimal | None] = mapped_column(NUMERIC, nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Links a protective stop to its entry; replays must return the original protective id.
    parent_order_id: Mapped[int | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(16))  # "open" | "filled" | "cancelled"
    client_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(String(36), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    node: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    action: Mapped[str | None] = mapped_column(String(8), nullable=True)
    detail: Mapped[dict] = mapped_column(JSONB)  # inputs, outputs, risk verdicts
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class NewsSentiment(Base):
    """One scored headline, written by the worker and read by the feature engine.

    Keyed by Alpaca's article id, so re-fetching a window never scores an article twice.
    symbols holds watchlist symbols only.
    """

    __tablename__ = "news_sentiment"

    article_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbols: Mapped[list[str]] = mapped_column(ARRAY(String(20)))
    headline: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(40))
    positive: Mapped[float] = mapped_column(Float)
    negative: Mapped[float] = mapped_column(Float)
    neutral: Mapped[float] = mapped_column(Float)
    model_revision: Mapped[str] = mapped_column(String(40))
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class JobRun(Base):
    """Last successful run of each background job: the worker's heartbeat."""

    __tablename__ = "job_runs"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_success_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
