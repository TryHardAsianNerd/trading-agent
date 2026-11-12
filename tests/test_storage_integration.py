"""Storage integration tests against the real compose TimescaleDB.

Skipped automatically when the database is unreachable (docker compose not up).
These are the only tests that need infrastructure; everything else is hermetic.
"""

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from trading_agent.config import Settings
from trading_agent.data.ingest import DEFAULT_BACKFILL_DAYS, ingest_daily_ohlcv
from trading_agent.data.models import Candle
from trading_agent.storage.db import init_db, make_engine, make_session_factory
from trading_agent.storage.models import CashFlow
from trading_agent.storage.repos import (
    CashFlowRepository,
    DecisionRepository,
    OhlcvRepository,
    SnapshotRepository,
    TradeRepository,
)


def _engine_or_skip():
    settings = Settings(_env_file=None)
    # connect_timeout keeps a docker-less run's skip fast instead of a 2-minute TCP wait
    engine = make_engine(settings.database_url, connect_args={"connect_timeout": 3})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("TimescaleDB not reachable — run: docker compose up -d")
    return engine


@pytest.fixture(scope="module")
def engine():
    eng = _engine_or_skip()
    init_db(eng)
    yield eng
    # Leave no test junk behind: §5's feature engine reads this same database.
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM ohlcv WHERE source = 'test'"))
        conn.execute(text("DELETE FROM trades WHERE broker = 'test'"))
        conn.execute(text("DELETE FROM portfolio_snapshots WHERE trading_mode LIKE 't-%'"))
        conn.execute(text("DELETE FROM decisions WHERE node = 'risk_manager' AND symbol = 'AAPL'"))


@pytest.fixture()
def session_factory(engine):
    return make_session_factory(engine)


def _candle(ts: datetime, close: str, symbol: str = "TEST-SYM") -> Candle:
    return Candle(
        symbol=symbol,
        asset_class="stock",
        ts=ts,
        open=Decimal("1.0"),
        high=Decimal("2.0"),
        low=Decimal("0.5"),
        close=Decimal(close),
        volume=Decimal("100"),
        source="test",
    )


def test_ohlcv_is_hypertable(engine):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT count(*) FROM timescaledb_information.hypertables "
                "WHERE hypertable_name = 'ohlcv'"
            )
        ).scalar()
    assert row == 1


def test_upsert_inserts_then_updates(session_factory):
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    symbol = f"T-{uuid.uuid4().hex[:8].upper()}"
    with session_factory() as session:
        repo = OhlcvRepository(session)
        assert repo.upsert_candles([_candle(ts, "2.0", symbol)]) == 1
        session.commit()
        # Same key, corrected close: must update, not raise or duplicate.
        assert repo.upsert_candles([_candle(ts, "3.5", symbol)]) == 1
        session.commit()

    with session_factory() as session:
        from trading_agent.storage.models import Ohlcv

        rows = session.query(Ohlcv).filter(Ohlcv.symbol == symbol).all()
        assert len(rows) == 1
        assert rows[0].close == Decimal("3.5")


def test_recent_daily_candles_excludes_test_source(session_factory):
    """Production candle reads must never see integration-test artifacts:
    source='test' rows are filtered; other sources are returned ascending."""
    symbol = f"T-{uuid.uuid4().hex[:8].upper()}"
    ts = datetime(2022, 1, 1, tzinfo=UTC)
    real = Candle(
        symbol=symbol,
        asset_class="stock",
        ts=ts,
        open=Decimal(1),
        high=Decimal(2),
        low=Decimal("0.5"),
        close=Decimal(1),
        volume=Decimal(1),
        source="itest",
    )
    with session_factory() as session:
        repo = OhlcvRepository(session)
        repo.upsert_candles([_candle(ts, "9.9", symbol), real])  # one 'test', one 'itest'
        session.commit()
    with session_factory() as session:
        candles = OhlcvRepository(session).recent_daily_candles(symbol)
    assert [c.source for c in candles] == ["itest"]
    with session_factory() as session:  # cleanup: 'itest' not covered by fixture here
        session.execute(text("DELETE FROM ohlcv WHERE source = 'itest'"))
        session.commit()


def test_recent_daily_candles_keep_one_bar_per_day_preferring_alpaca(session_factory):
    """D-046 moved crypto bars from CoinGecko to Alpaca; the table's key includes the
    source, so both providers' bars for the same days sit side by side. A reader that
    returned both would count each day twice (a "200-day" line over ~100 days)."""
    symbol = f"T-{uuid.uuid4().hex[:8].upper()}"
    days = [datetime(2022, 1, d, tzinfo=UTC) for d in (1, 2, 3)]

    def bar(ts: datetime, source: str, close: str) -> Candle:
        """One bar for this test's symbol."""
        return _candle(ts, close, symbol).model_copy(update={"source": source})

    with session_factory() as session:
        OhlcvRepository(session).upsert_candles(
            [
                bar(days[0], "coingecko", "1"),
                bar(days[1], "coingecko", "2"),
                bar(days[1], "alpaca", "20"),
                bar(days[2], "alpaca", "30"),
            ]
        )
        session.commit()
    try:
        with session_factory() as session:
            repo = OhlcvRepository(session)
            candles = repo.recent_daily_candles(symbol)
            latest_two = repo.recent_daily_candles(symbol, limit=2)
        assert [(c.ts, c.source, c.close) for c in candles] == [
            (days[0], "coingecko", Decimal(1)),  # the only bar that day
            (days[1], "alpaca", Decimal(20)),  # both providers: Alpaca's wins
            (days[2], "alpaca", Decimal(30)),
        ]
        assert [c.ts for c in latest_two] == days[1:]  # the limit counts days, not rows
    finally:
        with session_factory() as session:
            session.execute(text("DELETE FROM ohlcv WHERE symbol = :s"), {"s": symbol})
            session.commit()


def test_alpaca_ingest_flows_through_recent_daily_candles(session_factory):
    """Stock candles ingested via the Alpaca path (fake client, source 'itest'
    per the repo convention) land in ohlcv through the per-symbol ingest
    transaction and come back ascending from recent_daily_candles. A second
    cycle re-fetches the full window and its (split-adjusted) history
    overwrites the stored rows — the self-healing property D-021 relies on."""
    symbol = f"T-{uuid.uuid4().hex[:8].upper()}"
    now = datetime(2026, 9, 21, 15, 0, tzinfo=UTC)
    days = [datetime(2026, 9, 16, tzinfo=UTC) + timedelta(days=i) for i in range(3)]
    calls: list[tuple[datetime, datetime]] = []

    class FakeAlpaca:
        def __init__(self) -> None:
            self.scale = Decimal(1)

        def get_daily_candles(self, sym, start, end):
            calls.append((start, end))
            return [
                Candle(
                    symbol=sym,
                    asset_class="stock",
                    ts=ts,
                    open=Decimal("10") * self.scale,
                    high=Decimal("12") * self.scale,
                    low=Decimal("9") * self.scale,
                    close=Decimal(11 + i) * self.scale,
                    volume=Decimal("100"),
                    source="itest",
                )
                for i, ts in enumerate(days)
            ]

    settings = Settings(_env_file=None, stock_watchlist=symbol, crypto_watchlist="")
    window = (now - timedelta(days=DEFAULT_BACKFILL_DAYS), now)
    alpaca = FakeAlpaca()
    try:
        result = ingest_daily_ohlcv(settings, session_factory, alpaca=alpaca, now=now)
        assert result == {symbol: 3}
        assert calls == [window]
        with session_factory() as session:
            candles = OhlcvRepository(session).recent_daily_candles(symbol)
        assert [c.ts for c in candles] == days
        assert [c.close for c in candles] == [Decimal("11"), Decimal("12"), Decimal("13")]
        assert {c.source for c in candles} == {"itest"}

        alpaca.scale = Decimal("0.1")  # a 10:1 split: history comes back adjusted
        ingest_daily_ohlcv(settings, session_factory, alpaca=alpaca, now=now)
        assert calls == [window, window]  # full window again, not from the latest bar
        with session_factory() as session:
            candles = OhlcvRepository(session).recent_daily_candles(symbol)
        assert [c.close for c in candles] == [Decimal("1.1"), Decimal("1.2"), Decimal("1.3")]
    finally:
        with session_factory() as session:
            session.execute(text("DELETE FROM ohlcv WHERE symbol = :s"), {"s": symbol})
            session.commit()


def test_decision_log_roundtrip(session_factory):
    cycle_id = str(uuid.uuid4())
    with session_factory() as session:
        DecisionRepository(session).log(
            cycle_id=cycle_id,
            ts=datetime.now(UTC),
            node="risk_manager",
            detail={"verdict": "rejected", "reason": "kill_switch_active"},
            symbol="AAPL",
            action="buy",
        )
        session.commit()

    with session_factory() as session:
        from trading_agent.storage.models import Decision

        row = session.query(Decision).filter(Decision.cycle_id == cycle_id).one()
        assert row.detail["reason"] == "kill_switch_active"
        assert row.node == "risk_manager"


def _snap(repo, ts, equity, mode):
    repo.record(
        ts=ts,
        equity=Decimal(equity),
        cash=Decimal(equity),
        positions={},
        peak_equity=Decimal(equity),
        trading_mode=mode,
    )


def test_equity_at_prefers_latest_snapshot_at_or_before_boundary(session_factory):
    """The anchor is equity AT the boundary (latest <= boundary), so overnight/downtime
    losses are visible to the daily breaker — NOT the first observation after the boundary."""
    mode = f"t-{uuid.uuid4().hex[:6]}"
    boundary = datetime(2024, 3, 5, tzinfo=UTC)
    with session_factory() as session:
        repo = SnapshotRepository(session)
        _snap(repo, boundary - timedelta(hours=2), "100000", mode)  # 22:00 prior day
        _snap(repo, boundary + timedelta(hours=8), "90000", mode)  # 08:00, post-crash
        session.commit()
    with session_factory() as session:
        assert SnapshotRepository(session).snapshot_at(boundary, mode).equity == Decimal(100000)


def test_equity_at_falls_back_to_earliest_after_when_nothing_precedes(session_factory):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    boundary = datetime(2024, 3, 5, tzinfo=UTC)
    with session_factory() as session:
        repo = SnapshotRepository(session)
        _snap(repo, boundary + timedelta(hours=8), "90000", mode)
        _snap(repo, boundary + timedelta(hours=9), "95000", mode)
        session.commit()
    with session_factory() as session:
        repo = SnapshotRepository(session)
        assert repo.snapshot_at(boundary, mode).equity == Decimal(90000)
        assert repo.snapshot_at(boundary, f"t-{uuid.uuid4().hex[:6]}") is None  # mode-scoped
        # A ledger that started after the 08:00 snapshot anchors on the first one after it.
        since = boundary + timedelta(hours=8, minutes=30)
        assert repo.snapshot_at(boundary, mode, since).equity == Decimal(95000)


def test_peak_equity_is_mode_scoped(session_factory):
    paper_mode = f"t-{uuid.uuid4().hex[:6]}"
    live_mode = f"t-{uuid.uuid4().hex[:6]}"
    with session_factory() as session:
        repo = SnapshotRepository(session)
        _snap(repo, datetime(2024, 1, 1, tzinfo=UTC), "100000", paper_mode)
        _snap(repo, datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=1), "500", live_mode)
        session.commit()
    with session_factory() as session:
        repo = SnapshotRepository(session)
        # A paper peak must never arm the live drawdown kill switch.
        assert repo.latest(live_mode).peak_equity == Decimal(500)
        assert repo.latest(paper_mode).peak_equity == Decimal(100000)


def test_snapshot_same_ts_updates_instead_of_colliding(session_factory):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    ts = datetime(2024, 6, 1, tzinfo=UTC)
    with session_factory() as session:
        repo = SnapshotRepository(session)
        _snap(repo, ts, "100000", mode)
        session.commit()
        _snap(repo, ts, "99000", mode)  # same PK: merge, not IntegrityError
        session.commit()
    with session_factory() as session:
        assert SnapshotRepository(session).latest(mode).equity == Decimal(99000)


def _close(repo, at, pnl, mode, symbol="T-STREAK"):
    repo.record_fill(
        filled_at=at,
        symbol=symbol,
        asset_class="stock",
        side="sell",
        quantity=Decimal(1),
        fill_price=Decimal(100),
        broker="test",
        trading_mode=mode,
        realized_pnl=Decimal(pnl) if pnl is not None else None,
    )


def test_trailing_loss_streak_counts_window_and_mode(session_factory):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    now = datetime(2024, 6, 10, tzinfo=UTC)
    with session_factory() as session:
        repo = TradeRepository(session)
        _close(repo, now - timedelta(days=30), "-5", mode)  # outside window
        _close(repo, now - timedelta(days=3), "2", mode)  # profit breaks streak
        _close(repo, now - timedelta(days=2), "-1", mode)
        _close(repo, now - timedelta(days=1), "-1", mode)
        _close(repo, now - timedelta(days=1), None, mode)  # buy-side NULL ignored
        _close(repo, now - timedelta(days=2), "-9", f"t-{uuid.uuid4().hex[:6]}")  # other mode
        session.commit()
    with session_factory() as session:
        repo = TradeRepository(session)
        assert repo.trailing_loss_streak(mode, since=now - timedelta(days=7)) == 2
        # Window excludes everything: old streaks self-heal.
        assert repo.trailing_loss_streak(mode, since=now) == 0


def test_fills_roundtrip_mode_scoped(session_factory):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    with session_factory() as session:
        repo = TradeRepository(session)
        repo.record_fill(
            filled_at=datetime(2024, 6, 1, tzinfo=UTC),
            symbol=symbol,
            asset_class="crypto",
            side="buy",
            quantity=Decimal("0.5"),
            fill_price=Decimal("90000"),
            broker="test",
            trading_mode=mode,
        )
        session.commit()
    with session_factory() as session:
        repo = TradeRepository(session)
        (fill,) = repo.fills_for_symbol(symbol, mode)
        assert fill.quantity == Decimal("0.5")
        assert repo.fills_for_symbol(symbol, f"t-{uuid.uuid4().hex[:6]}") == []


def _cash_flow(ts, kind, amount, mode, scheduled_for=None):
    """An unsaved stock-venue ledger row in a throwaway trading mode."""
    return CashFlow(
        ts=ts,
        trading_mode=mode,
        asset_class="stock",
        kind=kind,
        amount=Decimal(amount),
        core_amount=Decimal(0),
        equity_before=Decimal(0),
        scheduled_for=scheduled_for,
    )


def test_the_ledger_reads_from_the_latest_baseline_and_never_books_a_date_twice(
    session_factory,
):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    with session_factory() as session:
        repo = CashFlowRepository(session)
        assert repo.ledger(mode) == []
        repo.add(
            [
                _cash_flow(t0, "baseline", "-100000", mode),
                _cash_flow(t0, "deposit", "1000", mode, t0.date()),
                _cash_flow(t0 + timedelta(days=30), "baseline", "-101000", mode),
                _cash_flow(t0 + timedelta(days=30), "deposit", "1000", mode, date(2026, 1, 31)),
            ]
        )
        session.commit()
    with session_factory() as session:
        rows = CashFlowRepository(session).ledger(mode)
        assert [(r.kind, r.amount) for r in rows] == [
            ("baseline", Decimal(-101000)),
            ("deposit", Decimal(1000)),
        ]
    with session_factory() as session:
        CashFlowRepository(session).add(
            [_cash_flow(t0 + timedelta(days=31), "deposit", "500", mode, date(2026, 1, 31))]
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_the_swing_streak_skips_core_closes(session_factory):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    now = datetime(2024, 6, 10, tzinfo=UTC)
    with session_factory() as session:
        repo = TradeRepository(session)
        _close(repo, now - timedelta(days=3), "-1", mode)
        _close(repo, now - timedelta(days=2), "5", mode, symbol="VOO")  # a core trail sell
        _close(repo, now - timedelta(days=1), "-1", mode)
        session.commit()
    with session_factory() as session:
        repo = TradeRepository(session)
        assert repo.trailing_loss_streak(mode, since=now - timedelta(days=7)) == 1
        streak = repo.trailing_loss_streak(mode, since=now - timedelta(days=7), exclude=["VOO"])
        assert streak == 2


def test_swing_columns_roundtrip_and_the_peak_index(session_factory):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    t0 = datetime(2024, 6, 10, tzinfo=UTC) + timedelta(microseconds=uuid.uuid4().int % 10**6)
    with session_factory() as session:
        repo = SnapshotRepository(session)
        for i, index in enumerate(["1", "1.3", "0.9"]):
            repo.record(
                ts=t0 + timedelta(hours=i),
                equity=Decimal(1000),
                cash=Decimal(1000),
                positions={},
                peak_equity=Decimal(1000),
                trading_mode=mode,
                swing=(Decimal(300), Decimal(index)),
            )
        session.commit()
    with session_factory() as session:
        repo = SnapshotRepository(session)
        assert repo.peak_swing_index(mode, t0) == Decimal("1.3")
        assert repo.peak_swing_index(mode, t0 + timedelta(hours=2)) == Decimal("0.9")
        assert repo.latest(mode).swing_equity == Decimal(300)


def test_entry_stop_is_the_current_positions_buy_stop(session_factory):
    """The latest market buy since the last filled sale: a cancelled buy and a closed
    position's buy never count, a lost one does, and a core buy has no stop."""
    from trading_agent.storage.repos import OrderRepository

    mode = f"t-{uuid.uuid4().hex[:6]}"
    t0 = datetime(2026, 10, 1, 14, tzinfo=UTC)

    def book(repo, minutes, side, status, stop=None, order_type="market", symbol="T-STOP"):
        """Book one order row."""
        repo.ensure(
            broker="test",
            broker_order_id=uuid.uuid4().hex,
            created_at=t0 + timedelta(minutes=minutes),
            symbol=symbol,
            asset_class="stock",
            side=side,
            quantity=Decimal("0.18"),
            order_type=order_type,
            stop_loss_price=None if stop is None else Decimal(stop),
            status=status,
            trading_mode=mode,
        )

    with session_factory() as session:
        repo = OrderRepository(session)
        assert repo.entry_stop("T-STOP", mode) is None  # nothing booked
        book(repo, 0, "buy", "filled", "90")  # the previous position
        book(repo, 1, "sell", "filled")  # closed
        assert repo.entry_stop("T-STOP", mode) is None  # its stop is not this one's
        book(repo, 2, "buy", "lost", "94")  # the current position, booked as lost
        book(repo, 3, "buy", "cancelled", "99")  # an unfilled DAY buy
        book(repo, 4, "sell", "open", "93", order_type="stop")  # not a sale
        assert repo.entry_stop("T-STOP", mode) == Decimal(94)
        book(repo, 5, "buy", "filled", symbol="VOO")  # a core buy carries no stop
        assert repo.entry_stop("VOO", mode) is None
        # Closed by hand at the venue (no sale booked), then bought again: the newer stop.
        book(repo, 6, "buy", "filled", "97")
        assert repo.entry_stop("T-STOP", mode) == Decimal(97)
        session.rollback()


def test_init_db_fills_in_the_core_part_of_rows_booked_before_it_existed(engine, session_factory):
    """Old rows get the split in force before the column: crypto all core, 60% of the
    whole deposit at the stock venue, baselines none. A second run changes nothing."""
    mode = f"t-{uuid.uuid4().hex[:6]}"
    day = date(2026, 10, 2)
    with engine.begin() as conn:
        # Nullable as on a database created before the column (a fresh one has NOT NULL).
        conn.execute(text("ALTER TABLE cash_flows ALTER COLUMN core_amount DROP NOT NULL"))
    with session_factory() as session:
        for kind, venue, amount in (
            ("baseline", "stock", "-100000"),
            ("deposit", "crypto", "100"),
            ("deposit", "stock", "900"),
        ):
            session.add(
                CashFlow(
                    ts=datetime.now(UTC),
                    trading_mode=mode,
                    asset_class=venue,
                    kind=kind,
                    amount=Decimal(amount),
                    core_amount=None,
                    equity_before=Decimal(0),
                    scheduled_for=None if kind == "baseline" else day,
                )
            )
        session.commit()

    def core_parts():
        """Each row's (kind, venue, core amount), in booking order."""
        with session_factory() as session:
            rows = CashFlowRepository(session).ledger(mode)
            return [(r.kind, r.asset_class, r.core_amount) for r in rows]

    init_db(engine)
    expected = [
        ("baseline", "stock", Decimal(0)),
        ("deposit", "crypto", Decimal(100)),
        ("deposit", "stock", Decimal(600)),
    ]
    assert core_parts() == expected
    init_db(engine)
    assert core_parts() == expected


def test_venue_fills_are_one_venues_fills_since_a_time(session_factory):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    start = datetime(2026, 10, 2, 13, tzinfo=UTC)
    with session_factory() as session:
        trades = TradeRepository(session)
        for at, symbol, venue in (
            (start - timedelta(hours=1), "AAPL", "stock"),  # before the ledger
            (start + timedelta(hours=2), "MSFT", "stock"),
            (start + timedelta(hours=1), "AAPL", "stock"),
            (start + timedelta(hours=1), "BTC-USD", "crypto"),
        ):
            trades.record_fill(
                filled_at=at,
                symbol=symbol,
                asset_class=venue,
                side="buy",
                quantity=Decimal(1),
                fill_price=Decimal(1),
                broker="test",
                trading_mode=mode,
            )
        session.commit()
        got = [t.symbol for t in trades.venue_fills("stock", mode, start)]
    assert got == ["AAPL", "MSFT"]


def test_the_ledger_start_bounds_a_symbols_fills(session_factory):
    mode = f"t-{uuid.uuid4().hex[:6]}"
    start = datetime(2026, 10, 2, 13, 35, tzinfo=UTC)
    with session_factory() as session:
        repo = TradeRepository(session)
        for at in (start - timedelta(days=20), start + timedelta(days=14)):
            repo.record_fill(
                filled_at=at,
                symbol="T-BASIS",
                asset_class="stock",
                side="buy",
                quantity=Decimal(1),
                fill_price=Decimal(100),
                broker="test",
                trading_mode=mode,
            )
        flows = CashFlowRepository(session)
        assert flows.start(mode) is None
        flows.add([_cash_flow(start, "baseline", "-100000", mode)])
        session.flush()
        assert flows.start(mode) == start
        assert len(repo.fills_for_symbol("T-BASIS", mode)) == 2
        (after,) = repo.fills_for_symbol("T-BASIS", mode, flows.start(mode))
        assert after.filled_at == start + timedelta(days=14)
        session.rollback()
