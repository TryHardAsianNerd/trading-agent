"""Node 5, risk manager: portfolio snapshot, stop checks, and the deterministic gate.

Every cycle writes one decisions row, even when the portfolio cannot be assembled
(then everything is rejected). A proposal on a symbol whose stop breached this cycle
is rejected before the gate, so a close and a new order never both reach the
executor. With a paper ledger the gate sees only the swing sleeve (D-042).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from trading_agent.data.models import Candle
from trading_agent.data.providers.base import ProviderError
from trading_agent.graph.nodes.trade_executor import reconcile_open_orders
from trading_agent.graph.state import AgentState, ApprovedOrder, TradeProposal
from trading_agent.log import get_logger
from trading_agent.portfolio import core, swing
from trading_agent.portfolio.tracker import PortfolioTracker, StopBreach, day_start, week_start
from trading_agent.risk import macro_calendar
from trading_agent.risk.breakers import drawdown_pct
from trading_agent.risk.gate import evaluate
from trading_agent.risk.killswitch import stops_everything
from trading_agent.risk.limits import DEFAULT_LIMITS
from trading_agent.risk.models import MarketEnvironment, PortfolioState, RejectReason
from trading_agent.storage.models import CashFlow, Order, Trade
from trading_agent.storage.repos import (
    CashFlowRepository,
    DecisionRepository,
    OhlcvRepository,
    OrderRepository,
    SnapshotRepository,
    TradeRepository,
)

if TYPE_CHECKING:
    from trading_agent.graph.deps import CycleDeps

log = get_logger(__name__)

# Alert only (D-047): the breakers watch only the swing sleeve; this watches the account.
ACCOUNT_DRAWDOWN_ALERT_PCT = Decimal("0.15")


def apply_risk_gate(state: AgentState, *, deps: CycleDeps) -> dict:
    """Assemble the portfolio, snapshot it, check stops, gate proposals and audit the cycle.

    A swing-sleeve failure rejects only swing proposals; the core's orders and every
    stop sale still go out.
    """
    now = datetime.now(UTC)
    proposals: list[TradeProposal] = state.get("proposals") or []
    market = state.get("market_data") or {}
    prices = market.get("prices") or {}

    breaches: list[StopBreach] = []
    try:
        booked: list[dict[str, object]] = []
        failed: set[str] = set()
        if _ledger_running(deps):
            # Book last cycle's fills BEFORE measuring the sleeves: an unbooked core fill
            # would move money between the core's claim and swing cash (fake returns).
            booked, failed = reconcile_open_orders(deps, deps.tracker.trading_mode)
        portfolio = deps.tracker.build_state(prices, now, book_deposits=True)
        # Stops first: a failure later in this step must not hold back a stop sale.
        swing_held = {s for s in portfolio.positions if not core.is_core(s)}
        breaches = deps.tracker.check_swing_stops(portfolio, prices, swing_held)
        if not failed:  # an unreconciled venue's book would read as a mismatch
            try:
                deps.tracker.check_positions(portfolio)
            except Exception:
                log.warning("position_check_failed", exc_info=True)

        symbols = set(prices) | set(portfolio.positions) | {p.symbol for p in proposals}
        with deps.session_factory() as session:
            repo = OhlcvRepository(session)
            candles_by_symbol = {s: repo.recent_daily_candles(s) for s in symbols}
        breaches += deps.tracker.check_trailing_stops(portfolio, candles_by_symbol, prices)
        breaches = list({b.symbol: b for b in reversed(breaches)}.values())  # one sale per symbol

        book = _core_book(deps)
        # Before this cycle's snapshot: the newest one is the previous cycle's.
        _alert_account_drawdown(deps, portfolio, book.flows[0].ts if book else None)

        core_proposals = [
            p.symbol for p in proposals if p.action != "hold" and core.is_core(p.symbol)
        ]
        if core_proposals:
            log.warning("analyst_core_proposals_dropped", symbols=core_proposals)
        breached = {b.symbol for b in breaches}
        gated = [
            p
            for p in proposals
            if p.action == "hold" or (p.symbol not in breached and not core.is_core(p.symbol))
        ]
        preempted = [p for p in proposals if p.action != "hold" and p.symbol in breached]

        environment, env_audit = _environment(deps, now)
        swing_record = None
        try:
            if book is None:
                gate_view, limits, swing_audit = portfolio, DEFAULT_LIMITS, None
            else:
                # This cycle's price, else the newest stored bar - trusted only when this
                # cycle's ingest refreshed it (a stale close would fake a move, then undo it).
                ingested = market.get("ingested") or {}
                marks = {s: c[-1].close for s, c in candles_by_symbol.items() if c} | prices
                stale = sorted(
                    {b.name for b in deps.brokers if b.name in failed and b.asset_class == "stock"}
                    | {
                        s
                        for s in portfolio.positions
                        if not core.is_core(s) and s not in prices and not ingested.get(s)
                    }
                )
                gate_view, swing_audit, swing_record = _swing_view(
                    deps, portfolio, marks, book, now, bool(stale)
                )
                swing_audit.update(stale=stale, fills_booked=booked)
                limits = swing.SWING_LIMITS
            decision = evaluate(
                gated, gate_view, candles_by_symbol, prices, environment, deps.kill_switch, limits
            )
            swing_orders, gate_audit = decision.approved, _jsonable(decision.audit)
            gate_rejections = [
                {
                    "symbol": r.proposal.symbol,
                    "action": r.proposal.action,
                    "reason": r.reason.value,
                    "detail": r.detail,
                }
                for r in decision.rejected
            ]
        except Exception as e:  # a swing failure must not stop the core or a stop sale
            if book is None:  # no ledger: this gate covers the whole account
                raise
            log.exception("swing_measure_failed", cycle_id=state.get("cycle_id"))
            swing_orders, gate_audit, swing_record = [], {}, None
            swing_audit = {"error": type(e).__name__}
            gate_rejections = _reject_all(
                gated,
                RejectReason.EVALUATION_ERROR,
                f"swing sleeve not measured: {type(e).__name__}",
            )
        deps.tracker.snapshot(portfolio, now, swing=swing_record)
        try:
            core_orders, core_audit = _core(deps, portfolio, prices, now, book)
        except Exception as e:  # the core failing must not stop the swing sleeve's gate
            log.exception("core_plan_failed", cycle_id=state.get("cycle_id"))
            core_orders, core_audit = [], {"error": type(e).__name__}
    except Exception as e:  # fail closed, but never fail silent
        log.exception("risk_assembly_failed", cycle_id=state.get("cycle_id"))
        rejections = _reject_all(proposals, RejectReason.EVALUATION_ERROR, type(e).__name__)
        _write_audit(
            deps,
            state["cycle_id"],
            now,
            proposals,
            approved=[],
            rejections=rejections,
            breach_dicts=_breach_dicts(breaches),
            audit={"error": f"portfolio_assembly:{type(e).__name__}"},
            equity=None,
            cash=None,
        )
        return {
            "approved_orders": [],
            "rejections": rejections,
            "stop_breaches": _breach_dicts(breaches),  # found before the failure: still sold
            "portfolio_error": type(e).__name__,  # the scheduler's watchdog reads this
        }

    rejections = _reject_all(preempted, RejectReason.STOP_BREACH_PENDING, "position close pending")
    rejections += gate_rejections
    breach_dicts = _breach_dicts(breaches)

    approved = [
        {
            "symbol": o.symbol,
            "side": o.side,
            "quantity": str(o.quantity),
            "stop_loss_price": str(o.stop_loss_price),
        }
        for o in swing_orders + core_orders
    ]
    _write_audit(
        deps,
        state["cycle_id"],
        now,
        proposals,
        approved,
        rejections,
        breach_dicts,
        audit={
            **gate_audit,
            "environment": env_audit,
            "core": core_audit,
            "swing": swing_audit,
        },
        equity=str(portfolio.equity),
        cash=str(portfolio.cash),
    )

    return {
        "approved_orders": swing_orders + core_orders,
        "rejections": rejections,
        "stop_breaches": breach_dicts,
    }


def _alert_account_drawdown(
    deps: CycleDeps, portfolio: PortfolioState, since: datetime | None
) -> None:
    """Page once when the whole account crosses ACCOUNT_DRAWDOWN_ALERT_PCT below its peak.

    Nothing is paused: that would also stop the deposit buys and the core's trail sells.
    since is the paper ledger's start (older snapshots are another account). Never raises.
    """
    try:
        drawdown = drawdown_pct(portfolio)
        if drawdown < ACCOUNT_DRAWDOWN_ALERT_PCT:
            return
        with deps.session_factory() as session:
            prev = SnapshotRepository(session).latest(deps.tracker.trading_mode, since)
        was_below = (
            prev is not None
            and prev.peak_equity > 0
            and 1 - prev.equity / prev.peak_equity >= ACCOUNT_DRAWDOWN_ALERT_PCT
        )
        if was_below:
            return  # already below the line last cycle: paged then
        log.error(
            "account_drawdown_alert",
            reason=f"account {drawdown:.1%} below its peak; alert only, nothing paused",
        )
    except Exception:
        log.warning("account_drawdown_alert_failed", exc_info=True)


@dataclass
class _CoreBook:
    """What the core sleeve and the swing view read from the book each cycle."""

    flows: list[CashFlow]
    candles: dict[str, list[Candle]]
    fills: dict[str, list[Trade]]
    open_orders: list[Order]
    stock_fills: list[Trade]


def _core_book(deps: CycleDeps) -> _CoreBook | None:
    """Read the core's inputs, or None without a paper ledger (no core, no sleeves)."""
    mode = deps.tracker.trading_mode
    with deps.session_factory() as session:
        flows = CashFlowRepository(session).ledger(mode)
        if not flows:
            return None
        ohlcv, trades, orders = (
            OhlcvRepository(session),
            TradeRepository(session),
            OrderRepository(session),
        )
        return _CoreBook(
            flows=flows,
            candles={
                s: ohlcv.recent_daily_candles(s, limit=core.CANDLES_NEEDED)
                for s in core.CORE_TARGETS
            },
            fills={s: trades.fills_for_symbol(s, mode) for s in core.CORE_TARGETS},
            open_orders=[
                o
                for b in deps.brokers
                for o in orders.open_orders(b.name, mode)
                if core.is_core(o.symbol) and o.order_type == "market"
            ],
            stock_fills=trades.venue_fills("stock", mode, flows[0].ts),
        )


def _ledger_running(deps: CycleDeps) -> bool:
    """Whether a paper ledger is running for this trading mode."""
    with deps.session_factory() as session:
        return bool(CashFlowRepository(session).ledger(deps.tracker.trading_mode))


def _swing_view(
    deps: CycleDeps,
    portfolio: PortfolioState,
    marks: dict[str, Decimal],
    book: _CoreBook,
    now: datetime,
    stale: bool,
) -> tuple[PortfolioState, dict[str, object], tuple[Decimal, Decimal] | None]:
    """Measure the swing sleeve; return the gate's view, an audit dict, and (equity, index).

    marks are this cycle's prices, else the last complete closes. When stale (the stock
    book is unreconciled or a held mark is old), the last measurement stands, the sleeve
    gets no cash to buy with, and the snapshot record is None: an unbooked fill or a
    stale mark would otherwise read as a return.
    """
    mode = deps.tracker.trading_mode
    start = book.flows[0].ts
    stock_cash = portfolio.venue_cash.get("stock", Decimal(0))
    # Open core buys stay in the claim: their cash is still at the venue until they fill.
    claim = core.stock_core_cash(
        book.flows, [f for s in core.CORE_STOCKS for f in book.fills[s]], Decimal(0)
    )
    cash, equity = swing.equity(portfolio, marks, claim)
    with deps.session_factory() as session:
        prev = SnapshotRepository(session).latest_swing(mode, start)
    previous = (prev.swing_equity, prev.swing_index) if prev is not None else None
    new = [f for f in book.flows if f.kind == "deposit" and (prev is None or f.ts > prev.ts)]
    new_money = sum((f.amount - f.core_amount for f in new if f.asset_class == "stock"), Decimal(0))
    record: tuple[Decimal, Decimal] | None
    if stale and previous is not None:
        equity, index = previous  # measure again once the book is whole
        cash, record = Decimal(0), None
    else:
        index = swing.next_index(previous, equity, new_money)
        record = (equity, index)
    with deps.session_factory() as session:
        snaps = SnapshotRepository(session)
        day = snaps.snapshot_at(day_start(now), mode, start, swing_only=True)
        week = snaps.snapshot_at(week_start(now), mode, start, swing_only=True)
        stored_peak = snaps.peak_swing_index(mode, start)
        peak = index if stored_peak is None else max(stored_peak, index)
        window = now - timedelta(days=PortfolioTracker.LOSS_STREAK_WINDOW_DAYS)
        streak = TradeRepository(session).trailing_loss_streak(
            mode, since=max(window, start), exclude=list(core.CORE_TARGETS)
        )
    view = swing.view(
        portfolio,
        cash,
        equity,
        index,
        (day.swing_index if day else None, week.swing_index if week else None, peak),
        streak,
    )
    return (
        view,
        {
            "equity": str(equity),
            "cash": str(cash),
            "index": str(index),
            "core_claim": str(claim),
            "unexplained": str(swing.unexplained(stock_cash, book.flows, book.stock_fills)),
            "measured": str(record is not None),
        },
        record,
    )


def _core(
    deps: CycleDeps,
    portfolio: PortfolioState,
    prices: dict[str, Decimal],
    now: datetime,
    book: _CoreBook | None,
) -> tuple[list[ApprovedOrder], dict[str, object] | None]:
    """Plan the core sleeve's (orders, audit); none without a ledger or under a full halt.

    Breaker cooldowns and the drawdown kill switch stop only the swing sleeve (D-040).
    """
    if book is None:
        return [], None
    halted, reason = deps.kill_switch.is_halted()
    if stops_everything(halted, reason):
        return [], {"halted": reason}
    return core.plan(portfolio, prices, book.candles, book.fills, book.flows, book.open_orders, now)


def _environment(deps: CycleDeps, now: datetime) -> tuple[MarketEnvironment, dict[str, object]]:
    """Build this cycle's (MarketEnvironment, audit) from FRED's VIX and the macro calendar.

    A FRED failure warns and leaves VIX unknown: VIX only reduces size, so no halving.
    """
    vix = vix_date = None
    if deps.fred is not None:
        try:
            vix, vix_date = deps.fred.latest_vix()
        except ProviderError as e:
            # Sanitized text, safe to log: it tells a revoked key from a 429.
            log.warning("vix_unavailable", error_type=type(e).__name__, error=str(e))
        except Exception as e:  # unsanitized text may quote the key-bearing URL: type only
            log.warning("vix_unavailable", error_type=type(e).__name__)
        else:
            age = (now.date() - vix_date).days
            if age > DEFAULT_LIMITS.vix_max_age_days:
                # Kept, not dropped: dropping a stale close could only ever turn halving off.
                log.warning("vix_stale", vix_date=vix_date.isoformat(), age_days=age)
    stale = macro_calendar.stale_kinds(now, macro_calendar.EVENTS)
    if stale:
        log.warning("macro_calendar_stale", kinds=stale, note="no upcoming release on file")
    pause = macro_calendar.macro_pause(now, macro_calendar.EVENTS, DEFAULT_LIMITS)
    audit: dict[str, object] = {
        "vix": None if vix is None else str(vix),
        "vix_date": None if vix_date is None else vix_date.isoformat(),
        "macro_event_pause": pause,
    }
    return MarketEnvironment(vix=vix, macro_event_pause=pause), audit


def _breach_dicts(breaches: list[StopBreach]) -> list[dict[str, object]]:
    """Stop breaches as dicts with numerics as str (the state's audit shape)."""
    return [
        {k: str(v) if not isinstance(v, str) else v for k, v in b.model_dump().items()}
        for b in breaches
    ]


def _reject_all(
    proposals: list[TradeProposal], reason: RejectReason, detail: str
) -> list[dict[str, object]]:
    """One {symbol, action, reason, detail} rejection per non-hold proposal."""
    return [
        {"symbol": p.symbol, "action": p.action, "reason": reason.value, "detail": detail}
        for p in proposals
        if p.action != "hold"
    ]


def _write_audit(
    deps: CycleDeps,
    cycle_id: str,
    ts: datetime,
    proposals: list[TradeProposal],
    approved: list[dict],
    rejections: list[dict],
    breach_dicts: list[dict],
    audit: dict,
    equity: str | None,
    cash: str | None,
) -> None:
    """Write and commit the cycle's decisions row in a session of its own."""
    with deps.session_factory() as session:
        DecisionRepository(session).log(
            cycle_id=cycle_id,
            ts=ts,
            node="risk_manager",
            detail={
                # The weekly report filters on it (decisions has no mode column).
                "trading_mode": deps.tracker.trading_mode,
                "equity": equity,
                "cash": cash,
                "proposals": [
                    {
                        "symbol": p.symbol,
                        "action": p.action,
                        "confidence": p.confidence,
                        "reasoning": p.reasoning,
                    }
                    for p in proposals
                ],
                "approved": approved,
                "rejected": rejections,
                "stop_breaches": breach_dicts,
                "audit": audit,
            },
        )
        session.commit()


def _jsonable(value: object) -> object:
    """Recursively coerce audit metadata to JSON-safe types, stringifying the rest."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
