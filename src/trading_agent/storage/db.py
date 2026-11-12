"""Engine factory and idempotent schema bootstrap.

TimescaleDB is required: init_db raises rather than silently running without hypertables.
"""

from datetime import datetime

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from trading_agent.log import get_logger
from trading_agent.storage.models import Base

log = get_logger(__name__)


def make_engine(database_url: str, connect_args: dict | None = None) -> Engine:
    """Create an engine that validates pooled connections before use."""
    return create_engine(database_url, pool_pre_ping=True, connect_args=connect_args or {})


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """The app-wide session factory; objects stay usable after commit."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create the extension, tables and ohlcv hypertable; bring an existing schema up to date."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "SELECT create_hypertable('ohlcv', 'ts', "
                "if_not_exists => TRUE, migrate_data => TRUE)"
            )
        )
        # create_all never alters existing tables; keep pre-existing DBs current.
        conn.execute(
            text("ALTER TABLE trades ADD COLUMN IF NOT EXISTS realized_pnl NUMERIC(30, 10)")
        )
        conn.execute(
            text(
                "ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS "
                "trading_mode VARCHAR(8) NOT NULL DEFAULT 'paper'"
            )
        )
        for column in ("swing_equity", "swing_index"):
            conn.execute(
                text(
                    f"ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS {column} "
                    "NUMERIC(30, 10)"
                )
            )
        # DB-enforced PaperBroker idempotency; NULL keys stay distinct so stop rows are unaffected.
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_paper_orders_account_client "
                "ON paper_orders (account, client_order_id)"
            )
        )
        for column in (
            "filled_quantity NUMERIC(30, 10)",
            "filled_at TIMESTAMPTZ",
            "parent_order_id INTEGER",
        ):
            conn.execute(text(f"ALTER TABLE paper_orders ADD COLUMN IF NOT EXISTS {column}"))
        # Nullable so an agent on the old model can still book a deposit; rows without it get
        # the split they were booked under: crypto all core, stocks 60% of the whole deposit,
        # baselines none.
        conn.execute(text("ALTER TABLE cash_flows ADD COLUMN IF NOT EXISTS core_amount NUMERIC"))
        conn.execute(
            text(
                "UPDATE cash_flows c SET core_amount = CASE "
                "WHEN c.kind <> 'deposit' THEN 0 "
                "WHEN c.asset_class = 'crypto' THEN c.amount "
                "ELSE 0.60 * (SELECT sum(d.amount) FROM cash_flows d WHERE d.kind = 'deposit' "
                "AND d.trading_mode = c.trading_mode AND d.scheduled_for = c.scheduled_for) END "
                "WHERE c.core_amount IS NULL"
            )
        )
        conn.execute(text("ALTER TABLE orders ALTER COLUMN status TYPE VARCHAR(32)"))
        # Executor booking is idempotent per venue order id (resumed cycles).
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_orders_broker_order_id "
                "ON orders (broker, broker_order_id)"
            )
        )
    log.info("db_initialized", tables=sorted(Base.metadata.tables))


# LangGraph's checkpoint tables (created by PostgresSaver.setup, not by our models).
_CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")


def prune_checkpoints(engine: Engine, older_than: datetime) -> int:
    """Delete graph checkpoints of cycles whose decisions rows all predate older_than; return rows.

    A checkpoint's thread_id is its cycle_id, and the decisions rows give its age, so a cycle
    that crashed before the risk node is never pruned. Raises if the checkpoint tables are
    missing (before the first PostgresSaver.setup).
    """
    deleted = 0
    with engine.begin() as conn:
        for table in _CHECKPOINT_TABLES:
            result = conn.execute(
                text(
                    f"DELETE FROM {table} WHERE thread_id IN "
                    "(SELECT cycle_id FROM decisions GROUP BY cycle_id HAVING max(ts) < :cutoff)"
                ),
                {"cutoff": older_than},
            )
            deleted += result.rowcount
    log.info("checkpoints_pruned", rows=deleted, older_than=older_than.isoformat())
    return deleted
