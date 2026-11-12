"""Weekly performance report: the numbers, the message, the schedule."""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from trading_agent.config import Settings
from trading_agent.data.models import Candle
from trading_agent.reports.weekly import (
    WeeklyReport,
    _has_cash_flow,
    _max_drawdown,
    _steps,
    _time_weighted_exposure,
    build_report,
    format_report,
    last_week_start,
)
from trading_agent.storage.db import init_db, make_engine, make_session_factory
from trading_agent.storage.models import CashFlow, Decision, PortfolioSnapshot, Trade
from trading_agent.storage.repos import OhlcvRepository

MONDAY = datetime(2026, 9, 14, tzinfo=UTC)
# A week no production row can fall in: the DB test is isolated from real data.
PAST_WEEK = datetime(2001, 1, 1, tzinfo=UTC)  # a Monday


def test_last_week_start_is_the_previous_complete_iso_week():
    assert last_week_start(datetime(2026, 9, 21, 13, tzinfo=UTC)) == MONDAY  # Monday run
    assert last_week_start(datetime(2026, 9, 27, 23, tzinfo=UTC)) == MONDAY  # Sunday
    assert last_week_start(datetime(2026, 9, 28, 0, tzinfo=UTC)) == MONDAY + timedelta(days=7)


def test_max_drawdown_is_the_deepest_fall_from_a_running_high():
    assert _max_drawdown([Decimal(v) for v in (100, 120, 90, 110, 80)]) == pytest.approx(1 / 3)
    assert _max_drawdown([Decimal(100)]) == 0.0
    assert _max_drawdown([]) == 0.0


def _snap(hours: float, equity: int, cash: int) -> SimpleNamespace:
    """A snapshot-shaped object (ts, equity, cash) `hours` after MONDAY."""
    return SimpleNamespace(
        ts=MONDAY + timedelta(hours=hours), equity=Decimal(equity), cash=Decimal(cash)
    )


def test_a_move_the_positions_cannot_explain_is_a_cash_flow():
    """The crypto venue joining took the book from $10k to $110k with nothing
    invested - that must never read as a +1000% week."""
    assert _has_cash_flow(_steps([_snap(0, 10_000, 10_000), _snap(1, 110_000, 110_000)], []))
    # 50% invested: a 10% move is well within what the positions can do.
    assert not _has_cash_flow(_steps([_snap(0, 100_000, 50_000), _snap(1, 110_000, 50_000)], []))
    assert _has_cash_flow(_steps([_snap(0, 100_000, 90_000), _snap(1, 130_000, 90_000)], []))


def test_a_ledger_deposit_is_not_an_unexplained_move():
    """All cash, then a $500 ledger deposit: recorded money, not a cash-flow week."""
    deposit = SimpleNamespace(
        kind="deposit", amount=Decimal(500), ts=MONDAY + timedelta(minutes=30)
    )
    steps = _steps([_snap(0, 1_000, 1_000), _snap(1, 1_500, 1_500)], [deposit])
    assert steps[0][2] == Decimal(500)
    assert not _has_cash_flow(steps)
    # Booked at a snapshot's own time, the deposit is already inside that snapshot.
    at_first = SimpleNamespace(kind="deposit", amount=Decimal(500), ts=MONDAY)
    at_second = SimpleNamespace(kind="deposit", amount=Decimal(500), ts=MONDAY + timedelta(hours=1))
    assert _steps([_snap(0, 1_500, 1_500), _snap(1, 1_500, 1_500)], [at_first])[0][2] == 0
    assert _steps([_snap(0, 1_000, 1_000), _snap(1, 1_500, 1_500)], [at_second])[0][2] == 500


def test_exposure_is_weighted_by_how_long_each_snapshot_stood():
    """Six days fully in cash, one day 60% invested -> ~8.6%, not the 30%
    a per-snapshot mean would give."""
    week_end = MONDAY + timedelta(days=7)
    series = [_snap(0, 100_000, 100_000), _snap(24 * 6, 100_000, 40_000)]
    assert _time_weighted_exposure(series, MONDAY, week_end) == pytest.approx(0.6 / 7)
    assert _time_weighted_exposure([], MONDAY, week_end) == 0.0


def _report(**overrides) -> WeeklyReport:
    """A report with sensible defaults."""
    base = dict(
        week_start=MONDAY,
        start_equity=Decimal(100_000),
        end_equity=Decimal(101_500),
        cash_flow=False,
        snapshots=600,
        max_drawdown=0.02,
        mean_exposure=0.4,
        closed_trades=4,
        wins=3,
        realized_pnl=Decimal("1200.50"),
        gross_profit=Decimal(1500),
        gross_loss=Decimal("299.50"),
        benchmark_return=0.03,
        rejections={"position_cap_reached": 3, "weird_code": 1},
        failed_cycles=2,
        cycles=672,
    )
    base.update(overrides)
    if "growth" not in base:
        start, end = base["start_equity"], base["end_equity"]
        base["growth"] = float(end / start) if start and end is not None else None
    return WeeklyReport(**base)


def test_message_reads_plainly():
    lines = format_report(_report(), "paper").splitlines()
    assert lines[:3] == [
        "[paper] weekly report, week of 2026-09-14",
        "Account: +1.50% ($100,000 -> $101,500), worst dip 2.0%, 40% invested on average "
        "(600 snapshots)",
        "Closed trades: 4, won 3 (75%), net $1,200.50, profit factor 5.01",
    ]
    assert lines[3] == (
        "Stage-1 check: buy-and-hold of the watchlist made +3.00%; at the same 40% invested "
        "that is +1.20%, so the account is 0.30 pts ahead of it. One week is noise - "
        "stage 1 is judged over the whole paper period."
    )
    assert lines[4] == (
        "Cycles: 672 (2 could not read the accounts); trade ideas turned down: "
        "position already at its size limit x3, weird_code x1"
    )


def test_a_losing_week_is_named_even_when_ahead_of_the_benchmark():
    line = format_report(
        _report(end_equity=Decimal(99_000), benchmark_return=-0.05), "paper"
    ).splitlines()[3]
    assert "ahead of it. The account lost money this week." in line


def test_no_comparison_when_almost_nothing_was_invested():
    line = format_report(_report(mean_exposure=0.01), "paper").splitlines()[3]
    assert line.endswith("not comparable this week (almost nothing invested)")


def test_a_cash_flow_week_reports_no_return_and_no_verdict():
    lines = format_report(
        _report(start_equity=Decimal(10_000), end_equity=Decimal(110_000), cash_flow=True),
        "paper",
    ).splitlines()
    assert "money was added or removed" in lines[1] and "%" not in lines[1]
    assert lines[3].endswith("not comparable this week (almost nothing invested)")


def test_message_without_data_says_so():
    empty = _report(
        start_equity=None,
        end_equity=None,
        snapshots=0,
        closed_trades=0,
        wins=0,
        benchmark_return=None,
        rejections={},
        cycles=0,
    )
    lines = format_report(empty, "paper").splitlines()
    assert "was the agent running?" in lines[1]
    assert lines[2] == "Closed trades: none"
    assert lines[3] == "Buy-and-hold comparison: no watchlist prices for the week"
    assert lines[-1].endswith("trade ideas turned down: none")


# --- against the compose database, in a week no real row can reach ------------------


@pytest.fixture(scope="module")
def session_factory():
    """Sessions on the compose database; this module's rows removed afterwards."""
    engine = make_engine(Settings(_env_file=None).database_url, connect_args={"connect_timeout": 3})
    try:
        init_db(engine)
    except Exception:
        pytest.skip("TimescaleDB not reachable - run: docker compose up -d")
    yield make_session_factory(engine)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM portfolio_snapshots WHERE trading_mode LIKE 'w-%'"))
        conn.execute(text("DELETE FROM trades WHERE trading_mode LIKE 'w-%'"))
        conn.execute(text("DELETE FROM ohlcv WHERE source = 'itest' AND symbol LIKE 'WK%'"))
        conn.execute(text("DELETE FROM decisions WHERE cycle_id LIKE 'wk-%'"))
        conn.execute(text("DELETE FROM cash_flows WHERE trading_mode LIKE 'w-%'"))


def test_build_report_reads_the_week_and_nothing_outside_it(session_factory):
    mode = f"w-{uuid.uuid4().hex[:6]}"
    symbol, stale = (f"WK{uuid.uuid4().hex[:4].upper()}" for _ in range(2))
    week = PAST_WEEK
    with session_factory() as s:
        # 100k before the week, then 110k -> 99k -> 105k inside it, 200k after. Half
        # invested throughout, so every move is explainable by the positions.
        for ts, equity in [
            (week - timedelta(hours=1), 100_000),
            (week + timedelta(days=1), 110_000),
            (week + timedelta(days=3), 99_000),
            (week + timedelta(days=6), 105_000),
            (week + timedelta(days=8), 200_000),
        ]:
            s.add(
                PortfolioSnapshot(
                    ts=ts,
                    equity=Decimal(equity),
                    cash=Decimal(equity) / 2,
                    positions={},
                    peak_equity=Decimal(equity),
                    trading_mode=mode,
                )
            )
        for days, pnl in [(2, 300), (4, -100), (5, None), (9, 999)]:  # None = a buy
            s.add(
                Trade(
                    filled_at=week + timedelta(days=days),
                    symbol=symbol,
                    asset_class="stock",
                    side="sell" if pnl is not None else "buy",
                    quantity=Decimal(1),
                    fill_price=Decimal(100),
                    broker="test",
                    trading_mode=mode,
                    realized_pnl=Decimal(pnl) if pnl is not None else None,
                )
            )
        for i, detail in enumerate(
            [
                {"trading_mode": mode, "rejected": [{"reason": "illiquid"}] * 2},
                {"trading_mode": mode, "rejected": [], "audit": {"error": "portfolio:X"}},
                {"trading_mode": "other", "rejected": [{"reason": "illiquid"}]},  # not ours
            ]
        ):
            s.add(
                Decision(
                    cycle_id=f"wk-{mode}-{i}",
                    ts=week + timedelta(days=2, hours=i),
                    node="risk_manager",
                    detail=detail,
                )
            )
        # symbol: last close before the week 100 (a Friday), last inside it 110.
        # stale: no bar inside the week - left out, not counted as 0%.
        OhlcvRepository(s).upsert_candles(
            [
                Candle(
                    symbol=sym,
                    asset_class="stock",
                    ts=week + timedelta(days=d),
                    open=Decimal(1),
                    high=Decimal(600),
                    low=Decimal(1),
                    close=Decimal(close),
                    volume=Decimal(1),
                    source="itest",
                )
                for sym, d, close in [
                    (symbol, -3, 100),
                    (symbol, 4, 110),
                    (symbol, 7, 500),
                    (stale, -3, 100),
                ]
            ]
        )
        s.commit()

    with session_factory() as s:
        report = build_report(s, week, mode, [symbol, stale])

    assert (report.start_equity, report.end_equity) == (Decimal(100_000), Decimal(105_000))
    assert not report.cash_flow
    assert report.week_return == pytest.approx(0.05)
    assert report.max_drawdown == pytest.approx(0.1)  # 110k -> 99k
    assert report.mean_exposure == pytest.approx(0.5)
    assert report.snapshots == 3
    assert (report.closed_trades, report.wins) == (2, 1)
    assert report.realized_pnl == Decimal(200)
    assert (report.gross_profit, report.gross_loss) == (Decimal(300), Decimal(100))
    assert report.benchmark_return == pytest.approx(0.10)
    assert report.benchmark_missing == [stale]
    assert report.rejections == {"illiquid": 2}
    assert (report.cycles, report.failed_cycles) == (2, 1)


def test_the_ledger_takes_deposits_out_of_the_week_and_reports_them(session_factory):
    """Ledger started before the week at $1,000 (older history ignored); +10% by
    Wednesday, then a $500 deposit: the week is +10%, not +65%."""
    mode = f"w-{uuid.uuid4().hex[:6]}"
    week = PAST_WEEK + timedelta(days=7)
    start = week - timedelta(days=2)
    with session_factory() as s:
        for ts, equity in [
            (start - timedelta(days=1), 100_000),  # the paper venue before the ledger
            (start, 1_000),
            (week - timedelta(hours=1), 1_000),
            (week + timedelta(days=2), 1_100),
            (week + timedelta(days=3), 1_600),
        ]:
            s.add(
                PortfolioSnapshot(
                    ts=ts,
                    equity=Decimal(equity),
                    cash=Decimal(equity) / 2,
                    positions={},
                    peak_equity=Decimal(equity),
                    trading_mode=mode,
                )
            )
        for ts, kind, amount, before in [
            (start, "baseline", -100_000, 0),
            (start, "deposit", 1_000, 0),
            (week + timedelta(days=2, hours=12), "deposit", 500, 1_100),
        ]:
            s.add(
                CashFlow(
                    ts=ts,
                    trading_mode=mode,
                    asset_class="stock",
                    kind=kind,
                    amount=Decimal(amount),
                    core_amount=Decimal(0),
                    equity_before=Decimal(before),
                    scheduled_for=ts.date() if kind == "deposit" else None,
                )
            )
        s.commit()

    with session_factory() as s:
        report = build_report(s, week, mode, [])

    assert not report.cash_flow
    assert report.week_return == pytest.approx(0.10)
    assert report.deposits == Decimal(500)
    assert report.total_deposited == Decimal(1_500)
    assert report.ledger_start == start
    lines = format_report(report, mode).splitlines()
    assert lines[2] == "Deposits this week: $500 (not counted as gains)"
    assert lines[3] == (
        f"Since {start.date().isoformat()}: $1,500 deposited, now worth $1,600 (+$100)"
    )


def test_a_week_with_no_snapshots_has_no_return_even_with_last_weeks_anchor(
    session_factory,
):
    """Agent down all week: no return and no stage-1 verdict, as before the ledger."""
    mode = f"w-{uuid.uuid4().hex[:6]}"
    week = PAST_WEEK + timedelta(days=14)
    with session_factory() as s:
        s.add(
            PortfolioSnapshot(
                ts=week - timedelta(hours=1),
                equity=Decimal(1000),
                cash=Decimal(500),
                positions={},
                peak_equity=Decimal(1000),
                trading_mode=mode,
            )
        )
        s.commit()
    with session_factory() as s:
        report = build_report(s, week, mode, [])
    assert report.growth is None and report.week_return is None
