"""Weekly report: last week's account, trades and stage-1 check against buy-and-hold.

    python -m trading_agent.reports.weekly     # print last week's report

Ledger deposits are taken out of the return; any other equity change the positions cannot
explain makes the week's return not measurable. Past results only, not a forecast.
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise

from sqlalchemy.orm import Session, sessionmaker

from trading_agent.config import Settings, load_settings
from trading_agent.log import configure_logging
from trading_agent.storage.db import make_engine, make_session_factory
from trading_agent.storage.models import CashFlow, Decision, Ohlcv, PortfolioSnapshot, Trade
from trading_agent.storage.repos import CashFlowRepository

# Below this average invested share the benchmark comparison says nothing.
MIN_EXPOSURE_FOR_COMPARISON = 0.05
# A move bigger than the invested value (plus this slack) cannot have come from trading.
CASH_FLOW_SLACK = Decimal("0.001")
# How far before the week a benchmark start close may be (a long weekend + holiday).
BENCHMARK_START_MAX_AGE = timedelta(days=5)
# Plain words for the gate's most common rejection codes; others print as-is.
REASON_WORDS = {
    "position_cap_reached": "position already at its size limit",
    "insufficient_cash_after_reserve": "no cash left above the 20% reserve",
    "illiquid": "too thinly traded",
    "insufficient_data": "not enough price history",
    "no_position_to_sell": "sell idea with nothing held",
    "already_held": "buy idea for a position already held",
    "position_size_zero": "order too small",
    "daily_loss_breaker": "daily loss pause",
    "weekly_loss_breaker": "weekly loss pause",
    "consecutive_losses_breaker": "losing-streak pause",
    "breaker_cooldown": "during a pause",
    "kill_switch_active": "kill switch on",
    "max_drawdown_exceeded": "15% drop emergency stop",
    "macro_event_pause": "economic-news pause",
}


@dataclass(frozen=True)
class WeeklyReport:
    """One ISO week's numbers, Monday 00:00 to Monday 00:00 UTC."""

    week_start: datetime
    start_equity: Decimal | None  # last snapshot of the prior week, else the week's first
    end_equity: Decimal | None
    growth: float | None  # time-weighted growth factor, ledger deposits removed
    cash_flow: bool  # money moved outside the ledger, so the return is not measurable
    snapshots: int
    max_drawdown: float
    mean_exposure: float
    closed_trades: int  # fills with a realized P&L
    wins: int
    realized_pnl: Decimal
    gross_profit: Decimal
    gross_loss: Decimal  # positive
    benchmark_return: float | None
    benchmark_missing: list[str] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    failed_cycles: int = 0  # cycles whose portfolio could not be assembled
    cycles: int = 0
    deposits: Decimal = Decimal(0)
    ledger_start: datetime | None = None
    total_deposited: Decimal = Decimal(0)

    @property
    def week_return(self) -> float | None:
        """The week's time-weighted return, or None without snapshots or with a cash flow."""
        if self.cash_flow or self.growth is None:
            return None
        return self.growth - 1

    @property
    def matched_benchmark(self) -> float | None:
        """Buy-and-hold scaled to the account's average invested share."""
        if self.benchmark_return is None:
            return None
        return self.benchmark_return * self.mean_exposure


def last_week_start(now: datetime) -> datetime:
    """Monday 00:00 UTC opening the last complete ISO week."""
    this_monday = datetime.combine(
        (now - timedelta(days=now.weekday())).date(), datetime.min.time(), UTC
    )
    return this_monday - timedelta(days=7)


def build_report(
    session: Session, week_start: datetime, trading_mode: str, symbols: list[str]
) -> WeeklyReport:
    """Read one week's numbers from the database, scoped to the trading mode."""
    week_end = week_start + timedelta(days=7)
    flows = [f for f in CashFlowRepository(session).ledger(trading_mode) if f.ts < week_end]
    snapshot_query = session.query(PortfolioSnapshot).filter(
        PortfolioSnapshot.trading_mode == trading_mode,
        PortfolioSnapshot.equity > 0,  # snapshots taken before any venue was configured
    )
    if flows:  # history before the paper ledger started is another account
        snapshot_query = snapshot_query.filter(PortfolioSnapshot.ts >= flows[0].ts)
    snaps = (
        snapshot_query.filter(PortfolioSnapshot.ts >= week_start, PortfolioSnapshot.ts < week_end)
        .order_by(PortfolioSnapshot.ts)
        .all()
    )
    # The prior week's closing snapshot anchors the week - only if it IS last week's.
    before = (
        snapshot_query.filter(
            PortfolioSnapshot.ts < week_start,
            PortfolioSnapshot.ts >= week_start - timedelta(days=7),
        )
        .order_by(PortfolioSnapshot.ts.desc())
        .first()
    )
    series = ([before] if before else []) + snaps
    closes = (
        session.query(Trade)
        .filter(
            Trade.trading_mode == trading_mode,
            Trade.realized_pnl.isnot(None),
            Trade.filled_at >= week_start,
            Trade.filled_at < week_end,
        )
        .all()
    )
    risk_rows = [
        row
        for row in session.query(Decision)
        .filter(Decision.node == "risk_manager", Decision.ts >= week_start, Decision.ts < week_end)
        .all()
        # Rows written before the mode was recorded are counted (paper was the only mode).
        if (row.detail or {}).get("trading_mode", trading_mode) == trading_mode
    ]
    pnls = [t.realized_pnl for t in closes]
    benchmark, missing = _buy_and_hold(session, symbols, week_start, week_end)
    steps = _steps(series, flows)
    index = [Decimal(1)]
    for prev, nxt, deposited in steps:
        index.append(index[-1] * (nxt.equity - deposited) / prev.equity)
    return WeeklyReport(
        week_start=week_start,
        start_equity=series[0].equity if series else None,
        end_equity=snaps[-1].equity if snaps else None,
        growth=float(index[-1]) if snaps else None,
        cash_flow=_has_cash_flow(steps),
        snapshots=len(snaps),
        max_drawdown=_max_drawdown(index),
        mean_exposure=_time_weighted_exposure(series, week_start, week_end),
        closed_trades=len(pnls),
        wins=sum(1 for p in pnls if p > 0),
        realized_pnl=sum(pnls, Decimal(0)),
        gross_profit=sum((p for p in pnls if p > 0), Decimal(0)),
        gross_loss=-sum((p for p in pnls if p < 0), Decimal(0)),
        benchmark_return=benchmark,
        benchmark_missing=missing,
        rejections=dict(
            Counter(
                r["reason"] for row in risk_rows for r in (row.detail or {}).get("rejected") or []
            ).most_common()
        ),
        failed_cycles=sum(
            1 for row in risk_rows if ((row.detail or {}).get("audit") or {}).get("error")
        ),
        cycles=len(risk_rows),
        deposits=sum(
            (f.amount for f in flows if f.kind == "deposit" and f.ts >= week_start), Decimal(0)
        ),
        ledger_start=flows[0].ts if flows else None,
        total_deposited=sum((f.amount for f in flows if f.kind == "deposit"), Decimal(0)),
    )


def _steps(
    series: list[PortfolioSnapshot], flows: list[CashFlow]
) -> list[tuple[PortfolioSnapshot, PortfolioSnapshot, Decimal]]:
    """Consecutive snapshot pairs, each with the ledger deposits booked in (prev.ts, next.ts]."""
    return [
        (
            prev,
            nxt,
            sum(
                (f.amount for f in flows if f.kind == "deposit" and prev.ts < f.ts <= nxt.ts),
                Decimal(0),
            ),
        )
        for prev, nxt in pairwise(series)
    ]


def _has_cash_flow(steps: list[tuple[PortfolioSnapshot, PortfolioSnapshot, Decimal]]) -> bool:
    """Whether equity, net of deposits, ever moved by more than the invested value could."""
    for prev, nxt, deposited in steps:
        invested = prev.equity - prev.cash
        if abs(nxt.equity - deposited - prev.equity) > invested + prev.equity * CASH_FLOW_SLACK:
            return True
    return False


def _max_drawdown(equities: list[Decimal]) -> float:
    """Deepest fall from a running high, as a fraction (0.0 for under two values)."""
    peak, worst = None, 0.0
    for e in equities:
        peak = e if peak is None or e > peak else peak
        if peak:
            worst = max(worst, float((peak - e) / peak))
    return worst


def _time_weighted_exposure(
    series: list[PortfolioSnapshot], week_start: datetime, week_end: datetime
) -> float:
    """Invested share of equity over the week, each snapshot weighted by how long it stood."""
    total = weighted = 0.0
    for snap, nxt in zip(series, [*series[1:], None][: len(series)], strict=True):
        start = max(snap.ts, week_start)
        end = min(nxt.ts if nxt else week_end, week_end)
        hours = max((end - start).total_seconds(), 0.0)
        weighted += float(1 - snap.cash / snap.equity) * hours
        total += hours
    return weighted / total if total else 0.0


def _buy_and_hold(
    session: Session, symbols: list[str], week_start: datetime, week_end: datetime
) -> tuple[float | None, list[str]]:
    """Equal-weight buy-and-hold return of the watchlist over the week, and the symbols left out.

    A daily bar is stamped at its 00:00 UTC open, so the last bar before a boundary holds the
    latest close. A symbol without closes on both sides is left out rather than counted as 0%.
    """
    returns, missing = [], []
    for symbol in symbols:
        bars = (
            session.query(Ohlcv)
            .filter(
                Ohlcv.symbol == symbol,
                Ohlcv.resolution == "1d",
                Ohlcv.source != "test",
                Ohlcv.ts >= week_start - BENCHMARK_START_MAX_AGE,
                Ohlcv.ts < week_end,
            )
            .order_by(Ohlcv.ts)
            .all()
        )
        start = [b for b in bars if b.ts < week_start]
        end = [b for b in bars if b.ts >= week_start]
        if start and end:
            returns.append(float(end[-1].close / start[-1].close - 1))
        else:
            missing.append(symbol)
    return (sum(returns) / len(returns) if returns else None), missing


def format_report(report: WeeklyReport, trading_mode: str) -> str:
    """The Discord message, in plain language."""
    lines = [f"[{trading_mode}] weekly report, week of {report.week_start.date().isoformat()}"]
    ret = report.week_return
    if report.snapshots == 0:
        lines.append("Account: no account snapshots this week - was the agent running?")
    elif report.cash_flow:
        lines.append(
            f"Account: ${report.start_equity:,.0f} -> ${report.end_equity:,.0f}, but money was "
            "added or removed (a venue added or an account reset), so the week's return "
            "is not measurable"
        )
    else:
        lines.append(
            f"Account: {ret:+.2%} (${report.start_equity:,.0f} -> ${report.end_equity:,.0f}), "
            f"worst dip {report.max_drawdown:.1%}, {report.mean_exposure:.0%} invested on "
            f"average ({report.snapshots} snapshots)"
        )
    if report.deposits:
        lines.append(f"Deposits this week: ${report.deposits:,.0f} (not counted as gains)")
    if report.ledger_start and report.end_equity is not None:
        gain = report.end_equity - report.total_deposited
        sign = "+" if gain >= 0 else "-"
        lines.append(
            f"Since {report.ledger_start.date().isoformat()}: ${report.total_deposited:,.0f} "
            f"deposited, now worth ${report.end_equity:,.0f} ({sign}${abs(gain):,.0f})"
        )
    if report.closed_trades:
        if report.gross_loss:
            pf = f", profit factor {report.gross_profit / report.gross_loss:.2f}"
        else:
            pf = ", no losing trade" if report.gross_profit else ""
        lines.append(
            f"Closed trades: {report.closed_trades}, won {report.wins} "
            f"({report.wins / report.closed_trades:.0%}), net ${report.realized_pnl:,.2f}{pf}"
        )
    else:
        lines.append("Closed trades: none")
    lines.append(_comparison(report))
    reasons = ", ".join(
        f"{REASON_WORDS.get(k, k)} x{v}" for k, v in list(report.rejections.items())[:5]
    )
    lines.append(
        f"Cycles: {report.cycles} ({report.failed_cycles} could not read the accounts); "
        f"trade ideas turned down: {reasons or 'none'}"
    )
    return "\n".join(lines)


def _comparison(report: WeeklyReport) -> str:
    """The stage-1 line: the account against equal-exposure buy-and-hold, or why not."""
    if report.benchmark_return is None:
        return "Buy-and-hold comparison: no watchlist prices for the week"
    bench = f"buy-and-hold of the watchlist made {report.benchmark_return:+.2%}"
    if report.benchmark_missing:
        bench += f" (without {', '.join(report.benchmark_missing)})"
    ret = report.week_return
    if ret is None or report.mean_exposure < MIN_EXPOSURE_FOR_COMPARISON:
        return f"Stage-1 check: {bench}; not comparable this week (almost nothing invested)"
    gap = (ret - report.matched_benchmark) * 100
    side = "ahead of" if gap > 0 else "behind"
    lost = " The account lost money this week." if ret < 0 else ""
    return (
        f"Stage-1 check: {bench}; at the same {report.mean_exposure:.0%} invested that is "
        f"{report.matched_benchmark:+.2%}, so the account is {abs(gap):.2f} pts {side} it.{lost} "
        "One week is noise - stage 1 is judged over the whole paper period."
    )


def weekly_message(
    settings: Settings, session_factory: sessionmaker[Session], now: datetime
) -> str:
    """Build the message for the last complete week."""
    with session_factory() as session:
        report = build_report(
            session,
            last_week_start(now),
            settings.trading_mode,
            settings.stock_symbols + settings.crypto_symbols,
        )
    return format_report(report, settings.trading_mode)


def main(argv: list[str] | None = None) -> None:
    """CLI: print last week's report (does not post it)."""
    argparse.ArgumentParser(prog="trading_agent.reports.weekly").parse_args(argv)
    configure_logging()
    settings = load_settings()
    factory = make_session_factory(make_engine(settings.database_url))
    print(weekly_message(settings, factory, datetime.now(UTC)))


if __name__ == "__main__":
    main()
