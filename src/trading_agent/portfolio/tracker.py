"""Portfolio tracker: assembles risk.PortfolioState from the brokers and stored history.

Equity, cash and positions come from the brokers; anchors, peak and the loss streak are
derived from snapshots and fills, so no state lives in process memory. With a paper ledger
running, venue figures include its rows and every anchor is scaled by later deposits. Weeks
start Monday 00:00 UTC, as in risk.breakers.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from trading_agent.brokers.base import AccountSummary, Broker
from trading_agent.data.models import Candle
from trading_agent.log import get_logger
from trading_agent.portfolio import ledger
from trading_agent.risk.limits import DEFAULT_LIMITS, RiskLimits
from trading_agent.risk.models import PortfolioState, Position
from trading_agent.risk.stops import trailing_stop_price
from trading_agent.storage.models import CashFlow, Trade
from trading_agent.storage.repos import (
    CashFlowRepository,
    OrderRepository,
    SnapshotRepository,
    TradeRepository,
)

log = get_logger(__name__)

# 100x below the smallest real fill (a $10 order is 0.0001 BTC at $100k), above venue rounding.
POSITION_TOLERANCE = Decimal("0.000001")
# Cycles a mismatch must last before it pages: a sale filling mid-cycle shows for one.
MISMATCH_CYCLES = 2


class StopBreach(BaseModel):
    """An open position whose price has crossed its stop; the executor closes it in full."""

    symbol: str
    asset_class: str
    quantity: Decimal
    current_price: Decimal
    stop_price: Decimal


def average_cost_walk(fills: list[Trade]) -> tuple[Decimal, Decimal | None, datetime | None]:
    """Open quantity, average entry and entry time from one symbol's fills, oldest first.

    Average-cost method: a flat position resets the basis, and entry and time are None when
    the walk ends flat. Fees are left out of the basis.
    """
    qty = Decimal(0)
    avg: Decimal | None = None
    entry_time: datetime | None = None
    for fill in fills:
        if fill.side == "buy":
            if qty <= 0:
                qty, avg, entry_time = fill.quantity, fill.fill_price, fill.filled_at
            else:
                total_cost = avg * qty + fill.fill_price * fill.quantity
                qty += fill.quantity
                avg = total_cost / qty
        else:
            qty -= fill.quantity
            if qty <= 0:
                qty, avg, entry_time = Decimal(0), None, None
    return qty, avg, entry_time


def day_start(now: datetime) -> datetime:
    """UTC midnight of the current day."""
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def week_start(now: datetime) -> datetime:
    """Monday 00:00 UTC of the current ISO week."""
    midnight = day_start(now)
    return midnight - timedelta(days=midnight.weekday())


class PortfolioTracker:
    """Builds PortfolioState, persists snapshots, and detects stop breaches."""

    LOSS_STREAK_WINDOW_DAYS = 7  # old streaks self-heal; see trailing_loss_streak

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        brokers: list[Broker],
        trading_mode: str,
        limits: RiskLimits = DEFAULT_LIMITS,
    ) -> None:
        """Wire the tracker to storage and venues.

        Every history query is scoped to trading_mode, so paper history never feeds live
        anchors; the executor books fills under the same public attribute.
        """
        self._session_factory = session_factory
        self._brokers = brokers
        self.trading_mode = trading_mode
        self._limits = limits
        # Swing positions already reported at error level for having no stored stop.
        self._reported_without_stop: set[str] = set()
        # symbol -> consecutive cycles its app and broker quantities have disagreed.
        self._mismatch_cycles: dict[str, int] = {}

    def build_state(
        self,
        prices: dict[str, Decimal],
        now: datetime | None = None,
        book_deposits: bool = False,
    ) -> PortfolioState:
        """Assemble the risk gate's PortfolioState from venue data and history.

        `prices` is only the entry-price fallback. Only the risk node passes book_deposits,
        since it snapshots at the same `now` right after; a deposit inside an earlier-stamped
        snapshot would count twice in the peak. ValueError if a running paper ledger's
        venues differ from the configured ones.
        """
        now = now or datetime.now(UTC)
        equity = Decimal(0)
        cash = Decimal(0)
        venue_cash: dict[str, Decimal] = {}
        positions: dict[str, Position] = {}
        with self._session_factory() as session:
            trades = TradeRepository(session)
            snapshots = SnapshotRepository(session)
            untracked: dict[str, Decimal] = {}
            # A stop settling between the account and position reads understates cash;
            # corrects next cycle.
            accounts = [(broker, broker.get_account()) for broker in self._brokers]
            flows = CashFlowRepository(session).ledger(self.trading_mode)
            if flows:
                started = {f.asset_class for f in flows if f.kind == "baseline"}
                configured = {b.asset_class for b, _ in accounts}
                if started != configured:
                    raise ValueError(
                        f"venues {sorted(configured)} differ from the paper ledger's "
                        f"{sorted(started)}: restart the ledger"
                    )
                if book_deposits:
                    flows = self._book_due_deposits(session, flows, accounts, now)
            added = ledger.offsets(flows)
            since = flows[0].ts if flows else None
            for broker, summary in accounts:
                extra = added.get(broker.asset_class, Decimal(0))
                equity += summary.equity + extra
                cash += summary.cash + extra
                venue_cash[broker.asset_class] = (
                    venue_cash.get(broker.asset_class, Decimal(0)) + summary.cash + extra
                )
                for bp in broker.get_positions():
                    avg_entry = bp.avg_entry_price
                    if avg_entry is None:
                        _, avg_entry, _ = average_cost_walk(
                            trades.fills_for_symbol(bp.symbol, self.trading_mode, since)
                        )
                    if avg_entry is None:
                        avg_entry = prices.get(bp.symbol)
                        if avg_entry is None:
                            # Value stays in equity; the gate must see it as untracked.
                            untracked[bp.symbol] = bp.quantity
                            log.warning(
                                "position_untracked",
                                symbol=bp.symbol,
                                note="no lots, no trade history, no current price",
                            )
                            continue
                        log.warning("position_entry_fallback_to_current_price", symbol=bp.symbol)
                    positions[bp.symbol] = Position(
                        symbol=bp.symbol,
                        asset_class=bp.asset_class,
                        quantity=bp.quantity,
                        avg_entry_price=avg_entry,
                    )

            day_snap = snapshots.snapshot_at(day_start(now), self.trading_mode, since)
            week_snap = snapshots.snapshot_at(week_start(now), self.trading_mode, since)
            last = snapshots.latest(self.trading_mode, since)
            streak_since = now - timedelta(days=self.LOSS_STREAK_WINDOW_DAYS)
            streak = trades.trailing_loss_streak(
                self.trading_mode, since=max(streak_since, since) if since else streak_since
            )

        def anchored(value: Decimal, at: datetime) -> Decimal:
            """An anchor recorded at `at` with the deposits booked after it folded in."""
            return value * ledger.growth_after(flows, at)

        return PortfolioState(
            equity=equity,
            cash=cash,
            peak_equity=max(anchored(last.peak_equity, last.ts), equity) if last else equity,
            day_start_equity=anchored(day_snap.equity, day_snap.ts) if day_snap else equity,
            week_start_equity=anchored(week_snap.equity, week_snap.ts) if week_snap else equity,
            consecutive_losses=streak,
            positions=positions,
            untracked_positions=untracked,
            venue_cash=venue_cash,
        )

    def _book_due_deposits(
        self,
        session: Session,
        flows: list[CashFlow],
        accounts: list[tuple[Broker, AccountSummary]],
        now: datetime,
    ) -> list[CashFlow]:
        """Book and commit every paper deposit that has come due, and return the ledger re-read.

        A duplicate date is rolled back with a warning.
        """
        due = ledger.due_dates(flows, now.date())
        if not due:
            return flows
        added = ledger.offsets(flows)
        equity = sum(
            (s.equity + added.get(b.asset_class, Decimal(0)) for b, s in accounts), Decimal(0)
        )
        venues = {b.asset_class for b, _ in accounts}
        rows = []
        for day in due:
            rows += ledger.deposit_rows(ledger.DEPOSIT, venues, equity, now, day, self.trading_mode)
            equity += ledger.DEPOSIT
        repo = CashFlowRepository(session)
        repo.add(rows)
        try:
            session.commit()
            log.info("deposits_booked", dates=[d.isoformat() for d in due])
        except IntegrityError as e:
            session.rollback()
            log.warning("deposits_not_booked", error=type(e).__name__)
        return repo.ledger(self.trading_mode)

    def snapshot(
        self,
        state: PortfolioState,
        now: datetime | None = None,
        swing: tuple[Decimal, Decimal] | None = None,
    ) -> None:
        """Persist and commit the state as a snapshot, with (swing equity, index) under a ledger."""
        now = now or datetime.now(UTC)
        with self._session_factory() as session:
            SnapshotRepository(session).record(
                ts=now,
                equity=state.equity,
                cash=state.cash,
                positions={
                    s: {"qty": str(p.quantity), "avg_entry": str(p.avg_entry_price)}
                    for s, p in state.positions.items()
                },
                peak_equity=state.peak_equity,
                trading_mode=self.trading_mode,
                swing=swing,
            )
            session.commit()

    def check_swing_stops(
        self, state: PortfolioState, prices: dict[str, Decimal], symbols: set[str]
    ) -> list[StopBreach]:
        """The app-held stops: swing positions at or below the stop on their latest buy.

        A position with no price or no stored stop is skipped with a warning, never guessed
        safe or breached; a missing stop is also reported once at error level.
        """
        breaches = []
        self._reported_without_stop.intersection_update(symbols)
        with self._session_factory() as session:
            orders = OrderRepository(session)
            for symbol in sorted(symbols):
                position = state.positions[symbol]
                stop = orders.entry_stop(symbol, self.trading_mode)
                price = prices.get(symbol)
                if stop is None:
                    if symbol not in self._reported_without_stop:
                        self._reported_without_stop.add(symbol)
                        log.error("swing_position_without_stop", symbol=symbol)
                    else:
                        log.warning("swing_stop_unchecked", symbol=symbol, reason="no stored stop")
                    continue
                self._reported_without_stop.discard(symbol)
                if price is None:
                    log.warning("swing_stop_unchecked", symbol=symbol, reason="no price")
                    continue
                if price <= stop:
                    log.warning(
                        "swing_stop_breached", symbol=symbol, price=str(price), stop=str(stop)
                    )
                    breaches.append(
                        StopBreach(
                            symbol=symbol,
                            asset_class=position.asset_class,
                            quantity=position.quantity,
                            current_price=price,
                            stop_price=stop,
                        )
                    )
        return breaches

    def check_positions(self, state: PortfolioState) -> dict[str, tuple[Decimal, Decimal]]:
        """Compare the app's open quantity per symbol with the broker's; page a lasting mismatch.

        The app's quantity walks its own fills since the paper ledger's start; without a
        ledger nothing is compared. A mismatch pages (error level) once it has lasted
        MISMATCH_CYCLES cycles, then stays quiet until it clears. Read-only. Returns
        symbol -> (app quantity, broker quantity) for this cycle's mismatches.
        """
        with self._session_factory() as session:
            since = CashFlowRepository(session).start(self.trading_mode)
            if since is None:
                return {}
            trades = TradeRepository(session)
            fills: dict[str, list[Trade]] = {}
            for asset_class in sorted({b.asset_class for b in self._brokers}):
                for fill in trades.venue_fills(asset_class, self.trading_mode, since):
                    fills.setdefault(fill.symbol, []).append(fill)
        app = {symbol: average_cost_walk(f)[0] for symbol, f in fills.items()}
        broker = {s: p.quantity for s, p in state.positions.items()}
        broker |= state.untracked_positions
        mismatched = {
            s: (app.get(s, Decimal(0)), broker.get(s, Decimal(0)))
            for s in app.keys() | broker.keys()
            if abs(app.get(s, Decimal(0)) - broker.get(s, Decimal(0))) > POSITION_TOLERANCE
        }
        for symbol in [s for s in self._mismatch_cycles if s not in mismatched]:
            if self._mismatch_cycles.pop(symbol) >= MISMATCH_CYCLES:
                log.info("position_mismatch_cleared", symbol=symbol)
        for symbol, (app_qty, broker_qty) in sorted(mismatched.items()):
            self._mismatch_cycles[symbol] = self._mismatch_cycles.get(symbol, 0) + 1
            if self._mismatch_cycles[symbol] == MISMATCH_CYCLES:
                log.error(
                    "position_mismatch",
                    symbol=symbol,
                    app_qty=str(app_qty),
                    broker_qty=str(broker_qty),
                )
        return mismatched

    def check_trailing_stops(
        self,
        state: PortfolioState,
        candles_by_symbol: dict[str, list[Candle]],
        prices: dict[str, Decimal],
    ) -> list[StopBreach]:
        """Positions at or below the fixed-percent trail, when the limits set one (off by default).

        The trail sits trailing_stop_pct below the high since the entry day. A position with
        no price or no entry time in its fills is skipped with a warning: pre-entry highs
        would flag healthy positions bought below a prior high.
        """
        if self._limits.trailing_stop_pct is None:
            return []
        breaches = []
        with self._session_factory() as session:
            trades = TradeRepository(session)
            since = CashFlowRepository(session).start(self.trading_mode)
            for symbol, position in state.positions.items():
                price = prices.get(symbol)
                if price is None:
                    log.warning("trailing_stop_unevaluable", symbol=symbol, reason="no price")
                    continue
                _, _, entry_time = average_cost_walk(
                    trades.fills_for_symbol(symbol, self.trading_mode, since)
                )
                if entry_time is None:
                    log.warning(
                        "trailing_stop_unevaluable",
                        symbol=symbol,
                        reason="no fill history to establish entry time",
                    )
                    continue
                candles = candles_by_symbol.get(symbol, [])
                high_water = _high_since(candles, entry_time, position.avg_entry_price)
                stop = trailing_stop_price(high_water, self._limits)
                if price <= stop:
                    breaches.append(
                        StopBreach(
                            symbol=symbol,
                            asset_class=position.asset_class,
                            quantity=position.quantity,
                            current_price=price,
                            stop_price=stop,
                        )
                    )
                    log.warning(
                        "trailing_stop_breached",
                        symbol=symbol,
                        price=str(price),
                        stop=str(stop),
                    )
        return breaches


def _high_since(candles: list[Candle], entry_time: datetime, entry_price: Decimal) -> Decimal:
    """High-water mark since the entry day, floored at the entry price.

    Daily bars are stamped 00:00 UTC, so the entry day's own bar counts, including any spike
    before the entry: the conservative direction.
    """
    boundary = day_start(entry_time)
    return max([entry_price, *(c.high for c in candles if c.ts >= boundary)])
