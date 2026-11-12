"""The realized-P&L repair, against the compose database on throwaway trading modes."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text

from trading_agent.config import Settings
from trading_agent.portfolio import repair_pnl
from trading_agent.storage.db import init_db, make_engine, make_session_factory
from trading_agent.storage.models import CashFlow
from trading_agent.storage.repos import TradeRepository

START = datetime.now(UTC) - timedelta(days=10)


@pytest.fixture(scope="module")
def session_factory():
    engine = make_engine(Settings(_env_file=None).database_url, connect_args={"connect_timeout": 3})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("TimescaleDB not reachable - run: docker compose up -d")
    init_db(engine)
    yield make_session_factory(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM trades WHERE broker = 'test' AND trading_mode LIKE 't-%'"))
        conn.execute(text("DELETE FROM cash_flows WHERE trading_mode LIKE 't-%'"))


def _book(session, mode, symbol, side, qty, price, at, pnl=None):
    """Record one throwaway fill."""
    TradeRepository(session).record_fill(
        filled_at=at,
        symbol=symbol,
        asset_class="stock",
        side=side,
        quantity=Decimal(qty),
        fill_price=Decimal(price),
        broker="test",
        trading_mode=mode,
        realized_pnl=None if pnl is None else Decimal(pnl),
    )


def _ledger_with_a_blended_sale(session_factory):
    """A ledger whose AAPL sale was booked against a lot sold, unbooked, at its start."""
    mode = f"t-{uuid.uuid4().hex[:6]}"
    with session_factory() as session:
        _book(session, mode, "AAPL", "buy", "32", "200", START - timedelta(days=20))
        session.add(
            CashFlow(
                ts=START,
                trading_mode=mode,
                asset_class="stock",
                kind="baseline",
                amount=Decimal(-100000),
                core_amount=Decimal(0),
                equity_before=Decimal(0),
            )
        )
        _book(session, mode, "AAPL", "buy", "0.5", "100", START + timedelta(days=1))
        _book(session, mode, "AAPL", "sell", "0.5", "90", START + timedelta(days=2), pnl="-54.6")
        _book(session, mode, "MSFT", "buy", "1", "100", START + timedelta(days=1))
        _book(session, mode, "MSFT", "sell", "1", "110", START + timedelta(days=2), pnl="10")
        session.commit()
    return mode


def test_the_dry_run_lists_only_the_blended_sale_and_writes_nothing(session_factory):
    mode = _ledger_with_a_blended_sale(session_factory)
    with session_factory() as session:
        (change,) = repair_pnl.corrections(session, mode)
        session.rollback()
    assert (change.symbol, change.booked, change.fixed) == ("AAPL", Decimal("-54.6"), Decimal(-5))
    with session_factory() as session:
        assert len(repair_pnl.corrections(session, mode)) == 1


def test_write_corrects_the_row_and_a_second_run_finds_nothing(session_factory):
    mode = _ledger_with_a_blended_sale(session_factory)
    with session_factory() as session:
        repair_pnl.write(session, repair_pnl.corrections(session, mode))
        session.commit()
    with session_factory() as session:
        sale = TradeRepository(session).fills_for_symbol("AAPL", mode)[-1]
        assert sale.realized_pnl == Decimal(-5)
        assert repair_pnl.corrections(session, mode) == []


def test_without_a_ledger_there_is_nothing_to_correct(session_factory):
    with session_factory() as session:
        assert repair_pnl.corrections(session, f"t-{uuid.uuid4().hex[:6]}") == []


def test_the_cli_refuses_live_mode(monkeypatch):
    monkeypatch.setattr(repair_pnl, "load_settings", lambda: MagicMock(is_paper=False))
    monkeypatch.setattr(repair_pnl, "configure_logging", lambda: None)
    with pytest.raises(SystemExit, match="paper mode only"):
        repair_pnl.main([])
