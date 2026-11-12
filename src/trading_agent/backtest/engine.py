"""Daily event-driven backtest through the live risk gate, with the live limits (D-029).

The model's out-of-sample scores stand in for the analyst, since the LLM has read the
history it would trade. Each day: approved orders fill at the open, stops are checked
against the low, the book is marked at the close, then the gate decides. No VIX or macro
calendar, and breakers run on daily closes, so drawdown trips can come later than live.
"""

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

import pandas as pd

from trading_agent.brokers.paper import TAKER_FEE_PCT
from trading_agent.data.models import Candle
from trading_agent.graph.state import ApprovedOrder, TradeProposal
from trading_agent.portfolio.tracker import PortfolioTracker
from trading_agent.risk.gate import evaluate
from trading_agent.risk.killswitch import KILL_SWITCH_KEY, KillSwitch
from trading_agent.risk.limits import DEFAULT_LIMITS, RiskLimits
from trading_agent.risk.models import MarketEnvironment, PortfolioState, Position

# Decisions are taken at the US close; the simulated clock drives cooldown TTLs.
DECISION_TIME = time(20, 0)
# Enough bars for the gate's ATR-14 and 20-day liquidity window.
GATE_CANDLES = 60
# The live tracker's loss-streak window: only closes this recent count.
LOSS_STREAK_WINDOW = timedelta(days=PortfolioTracker.LOSS_STREAK_WINDOW_DAYS)
COINBASE_TAKER_FEE = TAKER_FEE_PCT


@dataclass(frozen=True)
class StopRule:
    """How a held position's trailing stop works; the ATR stop is fixed at entry.

    Set at most one of trail_pct and trail_atr. With trail_on_close the trail is tested
    against the close and exits at the next open, since a close cannot be traded once seen.
    """

    trail_pct: Decimal | None = None
    trail_atr: Decimal | None = None
    trail_on_close: bool = False

    def trail(self, high_water: Decimal, atr: Decimal) -> Decimal | None:
        """The trailing stop level below the high-water mark, or None when there is no trail."""
        if self.trail_pct is not None:
            return high_water * (1 - self.trail_pct)
        if self.trail_atr is not None:
            return high_water - self.trail_atr * atr
        return None


@dataclass(frozen=True)
class Params:
    """One strategy configuration.

    `edge` is how far the score must sit above (buy) or below (sell) the base rate; `stops`
    None uses the limits' own trail, as live does.
    """

    edge: float
    slippage: Decimal
    starting_cash: Decimal = Decimal(100_000)
    crypto_fee: Decimal = COINBASE_TAKER_FEE
    stops: StopRule | None = None

    def cost(self, asset_class: str) -> Decimal:
        """Fraction of price lost on one fill: slippage, plus the fee on crypto."""
        return self.slippage + (self.crypto_fee if asset_class == "crypto" else Decimal(0))


@dataclass
class ClosedTrade:
    """One round trip, with fill prices after costs; reason is "stop" or "signal"."""

    symbol: str
    entry_day: date
    exit_day: date
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    reason: str

    @property
    def pnl(self) -> Decimal:
        """Realized profit or loss."""
        return (self.exit_price - self.entry_price) * self.quantity


@dataclass
class Result:
    """What one simulation produced: end-of-day equity and invested fraction, by day."""

    equity: pd.Series
    exposure: pd.Series
    trades: list[ClosedTrade]
    halted_on: date | None = None
    rejections: dict[str, int] = field(default_factory=dict)


@dataclass
class _Holding:
    """A simulated open position."""

    quantity: Decimal
    entry_price: Decimal
    entry_day: date
    atr_stop: Decimal
    atr: Decimal
    high_water: Decimal
    asset_class: str


class SimRedis:
    """The slice of Redis KillSwitch uses, with TTLs on the simulated clock."""

    def __init__(self) -> None:
        """Start empty at the epoch."""
        self.now = datetime(1970, 1, 1, tzinfo=UTC)
        self._data: dict[str, tuple[str, datetime | None]] = {}

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        """Store a value, optionally expiring ex seconds after the current sim time."""
        expires = self.now + timedelta(seconds=ex) if ex is not None else None
        self._data[key] = (value, expires)

    def get(self, key: str) -> str | None:
        """Read a value, or None when absent or expired at the current sim time."""
        value, expires = self._data.get(key, (None, None))
        if value is None or (expires is not None and self.now >= expires):
            return None
        return value

    def delete(self, key: str) -> None:
        """Remove a key."""
        self._data.pop(key, None)


def backtest_limits(limits: RiskLimits = DEFAULT_LIMITS) -> RiskLimits:
    """The live limits with the crypto liquidity minimum zeroed.

    Alpaca's single-venue crypto volume is not the CoinGecko aggregate that minimum is
    calibrated on.
    """
    return replace(limits, min_avg_dollar_volume_crypto=Decimal(0))


def simulate(
    history: dict[str, list[Candle]],
    scores: pd.DataFrame,
    params: Params,
    limits: RiskLimits,
) -> Result:
    """Run one strategy configuration over the scored window.

    `scores` rows carry "ts", "symbol", "score" and "base_rate", the share of rising outcomes
    in that row's fold's training data. Bars before the first score only feed lookbacks.
    """
    candles = {s: sorted(cs, key=lambda c: c.ts) for s, cs in history.items()}
    bars = {s: {c.ts.date(): c for c in cs} for s, cs in candles.items()}
    bar_days = {s: [c.ts.date() for c in cs] for s, cs in candles.items()}
    rows = scores[["ts", "symbol", "score", "base_rate"]].itertuples(index=False)
    score_at = {(ts.date(), s): (float(v), float(b)) for ts, s, v, b in rows}
    first = scores["ts"].min().date()
    last = max(max(b) for b in bars.values())
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]

    redis = SimRedis()
    kill_switch = KillSwitch(redis)
    cash = params.starting_cash
    holdings: dict[str, _Holding] = {}
    # Approved orders waiting for their next open, with the signal-day close
    # (the gate's stop is close - N x ATR, so the entry ATR can be recovered).
    pending: dict[str, tuple[ApprovedOrder, Decimal]] = {}
    last_close: dict[str, Decimal] = {}
    trades: list[ClosedTrade] = []
    equity: dict[date, Decimal] = {}
    exposure: dict[date, float] = {}
    peak = day_start = week_start = params.starting_cash
    rejections: dict[str, int] = {}
    halted_on: date | None = None
    rule = params.stops or StopRule(trail_pct=limits.trailing_stop_pct)
    exit_at_open: set[str] = set()  # close-triggered trail breaches, filled next open

    def close_position(symbol: str, day: date, price: Decimal, reason: str) -> None:
        """Exit a holding in full at a cost-adjusted price and record the round trip."""
        nonlocal cash
        h = holdings.pop(symbol)
        cash += h.quantity * price
        trades.append(
            ClosedTrade(symbol, h.entry_day, day, h.quantity, h.entry_price, price, reason)
        )

    for day in days:
        today = {s: b[day] for s, b in bars.items() if day in b}
        if day.weekday() == 0:  # the live tracker's week anchor: last close before Monday
            week_start = equity.get(day - timedelta(days=1), week_start)

        # 1. Open: close-triggered trail exits, then yesterday's approved orders.
        for symbol in [s for s in exit_at_open if s in today and s in holdings]:
            exit_at_open.discard(symbol)
            cost = params.cost(holdings[symbol].asset_class)
            close_position(symbol, day, today[symbol].open * (1 - cost), "stop")
        for symbol in [s for s in pending if s in today]:
            order, signal_close = pending.pop(symbol)
            bar = today[symbol]
            if order.side == "sell":
                if symbol in holdings:
                    cost = params.cost(holdings[symbol].asset_class)
                    close_position(symbol, day, bar.open * (1 - cost), "signal")
                continue
            price = bar.open * (1 + params.cost(order.asset_class))
            quantity = min(order.quantity, cash / price)  # a gap up can outrun the reserved cash
            if quantity <= 0:
                continue
            cash -= quantity * price
            atr_stop = order.stop_loss_price or Decimal(0)
            atr = (signal_close - atr_stop) / limits.atr_multiplier
            holdings[symbol] = _Holding(
                quantity, price, day, atr_stop, atr, price, order.asset_class
            )

        # 2. Stops: the ATR stop, or a higher trail. The high-water mark moves after the
        # check, so a day's own high never ratchets its own stop.
        for symbol in [s for s in holdings if s in today and s not in exit_at_open]:
            h, bar = holdings[symbol], today[symbol]
            cost = params.cost(h.asset_class)
            trail = rule.trail(h.high_water, h.atr)
            intraday = h.atr_stop
            if trail is not None and not rule.trail_on_close:
                intraday = max(intraday, trail)
            if bar.low <= intraday:
                close_position(symbol, day, min(bar.open, intraday) * (1 - cost), "stop")
            elif rule.trail_on_close and trail is not None and bar.close <= trail:
                exit_at_open.add(symbol)
            else:
                h.high_water = max(h.high_water, bar.high)

        # 3. Close: mark to market.
        last_close.update({s: b.close for s, b in today.items()})
        invested = sum((h.quantity * last_close[s] for s, h in holdings.items()), Decimal(0))
        equity[day] = cash + invested
        exposure[day] = float(invested / equity[day])
        peak = max(peak, equity[day])
        portfolio = _portfolio(
            day, cash, holdings, equity[day], peak, day_start, week_start, trades
        )
        day_start = equity[day]

        # 4. Decision at the close, through the live gate.
        # The gate runs even with nothing proposed, as in the live cycle: that is where
        # the drawdown kill switch arms and the breakers start their cooldowns.
        proposals = _proposals(today, day, score_at, params.edge, holdings, pending)
        redis.now = datetime.combine(day, DECISION_TIME, UTC)
        decision = evaluate(
            proposals,
            portfolio,
            {s: _window(candles[s], bar_days[s], day) for s in today},
            {s: b.close for s, b in today.items()},
            MarketEnvironment(),
            kill_switch,
            limits,
            now=redis.now,
        )
        for order in decision.approved:
            pending[order.symbol] = (order, today[order.symbol].close)
        for rejection in decision.rejected:
            rejections[rejection.reason.value] = rejections.get(rejection.reason.value, 0) + 1
        if halted_on is None and redis.get(KILL_SWITCH_KEY) is not None:
            halted_on = day

    def series(values: dict) -> pd.Series:
        """Day -> value as an ascending, day-indexed float series."""
        return pd.Series({pd.Timestamp(d): float(v) for d, v in values.items()}).sort_index()

    return Result(series(equity), series(exposure), trades, halted_on, rejections)


def _window(candles: list[Candle], days: list[date], day: date) -> list[Candle]:
    """The last GATE_CANDLES bars up to and including `day`; `days` are the candles' dates."""
    end = bisect_right(days, day)
    return candles[max(0, end - GATE_CANDLES) : end]


def _proposals(
    today: dict[str, Candle],
    day: date,
    score_at: dict[tuple[date, str], tuple[float, float]],
    edge: float,
    holdings: dict[str, _Holding],
    pending: dict[str, tuple[ApprovedOrder, Decimal]],
) -> list[TradeProposal]:
    """Gate proposals: buy flat symbols scoring `edge` above base rate, sell held ones below.

    A symbol with an order still pending gets no second one.
    """
    out = []
    for symbol, bar in sorted(today.items()):
        scored = score_at.get((day, symbol))
        if scored is None or symbol in pending:
            continue
        score, base_rate = scored
        held = symbol in holdings
        if not held and score >= base_rate + edge:
            action = "buy"
        elif held and score <= base_rate - edge:
            action = "sell"
        else:
            continue
        out.append(
            TradeProposal(
                symbol=symbol,
                asset_class=bar.asset_class,
                action=action,
                confidence=score,
                reasoning=f"backtest: score {score:.3f} vs base rate {base_rate:.3f}",
            )
        )
    return out


def _portfolio(
    day: date,
    cash: Decimal,
    holdings: dict[str, _Holding],
    equity_now: Decimal,
    peak: Decimal,
    day_start: Decimal,
    week_start: Decimal,
    trades: list[ClosedTrade],
) -> PortfolioState:
    """The gate's view of the simulated book at today's close.

    The loss streak counts only closes inside the live tracker's window.
    """
    streak = 0
    for t in reversed(trades):
        if t.exit_day < day - LOSS_STREAK_WINDOW or t.pnl >= 0:
            break
        streak += 1
    return PortfolioState(
        equity=equity_now,
        cash=cash,
        peak_equity=peak,
        day_start_equity=day_start,
        week_start_equity=week_start,
        consecutive_losses=streak,
        positions={
            s: Position(
                symbol=s,
                asset_class=h.asset_class,
                quantity=h.quantity,
                avg_entry_price=h.entry_price,
            )
            for s, h in holdings.items()
        },
    )
