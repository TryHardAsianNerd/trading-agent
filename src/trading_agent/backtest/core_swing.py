"""Core + swing backtest with the ledger's recurring deposits, through the live risk gate.

`python -m trading_agent.backtest.core_swing [--sweep]` ranks each core stop and kill-switch
scope on a tune half and reports it on a validate half. The core uses portfolio.core's rules;
deposits scale every anchor, so they never read as gains. `--sweep` runs the chosen plan under
every SWEEP_CONFIGS entry instead. Past results only; nothing here predicts future returns.
"""

import argparse
import json
import logging
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from itertools import product

import pandas as pd

from trading_agent.backtest.engine import (
    DECISION_TIME,
    ClosedTrade,
    Params,
    SimRedis,
    _Holding,
    _portfolio,
    _proposals,
    _window,
    backtest_limits,
)
from trading_agent.backtest.metrics import (
    deposit_and_hold,
    money_weighted_return,
    time_weighted_drawdown,
)
from trading_agent.backtest.run import OUT_DIR
from trading_agent.backtest.study import SLIPPAGE, trend_scores
from trading_agent.config import load_settings
from trading_agent.data.models import Candle
from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.graph.state import ApprovedOrder
from trading_agent.log import configure_logging
from trading_agent.ml.train import fetch_history
from trading_agent.portfolio import core as live
from trading_agent.portfolio.core import (
    CORE_CRYPTO,
    CORE_SHARE,
    CORE_STOCKS,
    CORE_TARGETS,
    TREND_DAYS,
)
from trading_agent.portfolio.ledger import (
    DEPOSIT,
    DEPOSIT_EVERY,
    FIRST_DEPOSIT,
    core_split,
    split,
)
from trading_agent.risk.gate import evaluate
from trading_agent.risk.killswitch import KILL_SWITCH_KEY, KillSwitch
from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.models import MarketEnvironment
from trading_agent.risk.stops import wilder_atr

SWING_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO",
    "JPM", "V", "COST", "WMT", "LLY", "XOM",
]  # fmt: skip
# Measured on the swing sleeve's equity.
SWING_POSITION_CAP = Decimal("0.25")
SWING_RESERVE = Decimal("0.20")
SWING_WARMUP = 60  # the gate's candle window (engine.GATE_CANDLES)
TREND_EDGE = 0.25  # trend scores are 0/1 around a 0.5 base rate (study.trend_scores)
KILL_SCOPES = ("everything", "swing only")
# Swing cash kept out of VOO for rounding and fees; smaller parks and unparks wait.
SWEEP_BUFFER = Decimal(5)
SWEEP_MIN = Decimal(10)


@dataclass
class _CoreHolding:
    """A core position built from many deposit buys."""

    quantity: Decimal
    avg_cost: Decimal
    atr: Decimal  # ATR-14 at the latest buy
    high_close: Decimal  # highest close since the position opened


@dataclass(frozen=True)
class CoreStop:
    """When a core holding is sold, checked on each close; set at most one field.

    `atr` is ATRs below average cost, `trail` a fraction below the high close, and
    `below_trend` a close under the 200-day line (which also gates buys).
    """

    atr: Decimal | None = None
    trail: Decimal | None = None
    below_trend: bool = False

    def breached(self, close: Decimal, holding: _CoreHolding, line: Decimal | None) -> bool:
        """Whether today's close sells the holding; `line` is the 200-day average or None."""
        if self.atr is not None:
            return close <= holding.avg_cost - self.atr * holding.atr
        if self.trail is not None:
            return live.trail_breached(close, holding.avg_cost, [holding.high_close], self.trail)
        return self.below_trend and line is not None and close < line


CORE_STOPS = {
    "no stop": CoreStop(),
    "3x ATR": CoreStop(atr=Decimal(3)),
    "trail 15%": CoreStop(trail=Decimal("0.15")),
    "trail 25%": CoreStop(trail=Decimal("0.25")),
    "below 200-day": CoreStop(below_trend=True),
}


@dataclass
class BookResult:
    """What one core + swing simulation produced, with end-of-day curves by day.

    `unaffordable` counts approved swing buys the cash could not fund above the minimum order.
    """

    equity: pd.Series
    swing_equity: pd.Series
    deposits: dict[date, Decimal]
    swing_trades: list[ClosedTrade]
    core_exits: int = 0
    unaffordable: int = 0
    halted_on: date | None = None
    rejections: dict[str, int] = field(default_factory=dict)
    invested: pd.Series | None = None
    sweep_trades: int = 0


def core_buys(
    stock_claim: Decimal,
    venue_cash: dict[str, Decimal],
    values: dict[str, Decimal],
    eligible: set[str],
    equity: Decimal,
) -> dict[str, Decimal]:
    """The core's deposit buys for one close, by the live rules."""
    crypto_value = sum((values.get(s, Decimal(0)) for s in CORE_CRYPTO), Decimal(0))
    cash = live.budgets(stock_claim, venue_cash, crypto_value, equity)
    return live.venue_buys(cash, values, eligible)


def simulate_book(
    history: dict[str, list[Candle]],
    start: date,
    end: date,
    stop: CoreStop,
    kill_scope: str,
    limits: RiskLimits,
    slippage: Decimal = SLIPPAGE,
    sweep: bool = False,
    sleeve_breakers: bool = False,
    parked_out: bool = False,
) -> BookResult:
    """Run the core + swing plan with deposits from `start` to `end`.

    Core orders are decided on each close and fill at the next open. `kill_scope` "everything"
    also stops core buying. `sweep` drops the swing reserve and parks idle swing cash in VOO.
    `sleeve_breakers` applies the live rules (D-042): the gate sees the swing sleeve's own
    equity and index; otherwise breakers watch the whole account (D-039). `parked_out` leaves
    parked VOO's own moves out of the sleeve index.
    """
    candles = {s: sorted(cs, key=lambda c: c.ts) for s, cs in history.items()}
    bars = {s: {c.ts.date(): c for c in cs} for s, cs in candles.items()}
    bar_days = {s: [c.ts.date() for c in cs] for s, cs in candles.items()}
    lines = {
        s: {
            d: v
            for d, v in zip(bar_days[s], live.average_closes(candles[s]), strict=True)
            if v is not None
        }
        for s in CORE_TARGETS
    }
    swing_symbols = [s for s in SWING_SYMBOLS if s in candles]
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    window = pd.Series(pd.to_datetime(days).tz_localize(UTC))
    trend = trend_scores({s: candles[s] for s in swing_symbols}, window) if swing_symbols else None
    score_at = {}
    if trend is not None:
        rows = trend[["ts", "symbol", "score", "base_rate"]].itertuples(index=False)
        score_at = {(ts.date(), s): (float(v), float(b)) for ts, s, v, b in rows}
    params = Params(TREND_EDGE, slippage)

    def cost(symbol: str) -> Decimal:
        """Fraction lost on one fill of a symbol (slippage, plus the fee on crypto)."""
        return params.cost(candles[symbol][0].asset_class)

    redis = SimRedis()
    kill_switch = KillSwitch(redis)
    stock_claim = crypto_cash = swing_cash = peak = day_start = week_start = Decimal(0)
    core: dict[str, _CoreHolding] = {}
    sold: dict[str, date] = {}  # stopped-out core symbol -> the day its sell filled
    pending_core: dict[str, tuple[Decimal, Decimal]] = {}  # symbol -> (reserved dollars, ATR)
    core_sells: set[str] = set()
    swing: dict[str, _Holding] = {}
    pending: dict[str, tuple[ApprovedOrder, Decimal]] = {}
    last_close: dict[str, Decimal] = {}
    last_line: dict[str, Decimal] = {}
    deposits: dict[date, Decimal] = {}
    equity: dict[date, Decimal] = {}
    swing_equity: dict[date, Decimal] = {}
    trades: list[ClosedTrade] = []
    rejections: dict[str, int] = {}
    core_exits = unaffordable = sweep_trades = 0
    parked = Decimal(0)  # swing-parked VOO units, apart from the core's
    parked_before = park_flow = Decimal(0)  # yesterday's parked value; today's net parking
    # The swing index moves with the sleeve's returns, not with its deposits.
    index, index_by_day, index_peak = Decimal(1), {}, Decimal(1)
    index_week = Decimal(1)
    swing_before: Decimal | None = None
    new_swing = Decimal(0)
    invested: dict[date, Decimal] = {}
    halted_on: date | None = None
    next_deposit = start

    def close_swing(symbol: str, day: date, price: Decimal, reason: str) -> None:
        """Exit a swing holding in full at a cost-adjusted price and record the round trip."""
        nonlocal swing_cash
        h = swing.pop(symbol)
        swing_cash += h.quantity * price
        trades.append(
            ClosedTrade(symbol, h.entry_day, day, h.quantity, h.entry_price, price, reason)
        )

    for day in days:
        today = {s: b[day] for s, b in bars.items() if day in b}
        if day.weekday() == 0:  # the live tracker's week anchor: last close before Monday
            week_start = equity.get(day - timedelta(days=1), week_start)
            index_week = index_by_day.get(day - timedelta(days=1), index_week)
        if day >= next_deposit:
            amount = FIRST_DEPOSIT if not deposits else DEPOSIT
            deposits[day] = amount
            next_deposit += DEPOSIT_EVERY
            venues = split(amount, {"stock", "crypto"})
            core_part = core_split(amount, {"stock", "crypto"})["stock"]
            crypto_cash += venues["crypto"]
            stock_claim += core_part
            swing_cash += venues["stock"] - core_part
            new_swing += venues["stock"] - core_part
            # Not a gain, and no dilution of a loss: scale every anchor by the deposit's share.
            scale = (day_start + amount) / day_start if day_start else None
            peak = peak * scale if scale else amount
            week_start = week_start * scale if scale else amount
            day_start += amount

        # 1. Open: core sells, core buys, then the swing orders the gate approved.
        for s in [s for s in core_sells if s in today]:
            core_sells.discard(s)
            h = core.pop(s)
            proceeds = h.quantity * today[s].open * (1 - cost(s))
            if s in CORE_CRYPTO:
                crypto_cash += proceeds
            else:
                stock_claim += proceeds
            sold[s] = day
            core_exits += 1
        for s in [s for s in pending_core if s in today]:
            dollars, atr = pending_core.pop(s)
            price = today[s].open * (1 + cost(s))
            quantity = dollars / price
            h = core.get(s)
            if h is None:
                core[s] = _CoreHolding(quantity, price, atr, Decimal(0))
            else:
                h.avg_cost = (h.quantity * h.avg_cost + dollars) / (h.quantity + quantity)
                h.quantity += quantity
                h.atr = atr
        for s in [s for s in pending if s in today]:
            order, signal_close = pending.pop(s)
            bar = today[s]
            if order.side == "sell":
                if s in swing:
                    close_swing(s, day, bar.open * (1 - cost(s)), "signal")
                continue
            price = bar.open * (1 + cost(s))
            need = order.quantity * price - swing_cash
            if sweep and need > 0 and parked > 0 and "VOO" in today:
                # Sell just enough parked VOO to fund the buy.
                unit = today["VOO"].open * (1 - cost("VOO"))
                units = min(parked, need / unit)
                parked -= units
                swing_cash += units * unit
                park_flow -= units * unit
                sweep_trades += 1
            shares = min(order.quantity, swing_cash / price)
            if shares * price < limits.min_order_value:
                unaffordable += 1
                continue
            swing_cash -= shares * price
            atr_stop = order.stop_loss_price or Decimal(0)
            atr = (signal_close - atr_stop) / limits.atr_multiplier
            swing[s] = _Holding(shares, price, day, atr_stop, atr, price, order.asset_class)

        if sweep and "VOO" in today and swing_cash - SWEEP_BUFFER >= SWEEP_MIN:
            spare = swing_cash - SWEEP_BUFFER
            parked += spare / (today["VOO"].open * (1 + cost("VOO")))
            swing_cash -= spare
            park_flow += spare
            sweep_trades += 1

        # 2. Swing stops at the day's low: optimistic, since live checks them every
        # 15 minutes in the session and not at all while the agent is down.
        for s in [s for s in swing if s in today]:
            h, bar = swing[s], today[s]
            if bar.low <= h.atr_stop:
                close_swing(s, day, min(bar.open, h.atr_stop) * (1 - cost(s)), "stop")

        # 3. Close: mark to market.
        last_close.update({s: b.close for s, b in today.items()})
        last_line.update({s: lines[s][day] for s in CORE_TARGETS if day in lines[s]})
        core_value = sum((h.quantity * last_close[s] for s, h in core.items()), Decimal(0))
        swing_value = sum((h.quantity * last_close[s] for s, h in swing.items()), Decimal(0))
        parked_value = parked * last_close.get("VOO", Decimal(0))
        reserved = sum((d for d, _ in pending_core.values()), Decimal(0))
        core_cash = stock_claim + crypto_cash
        total = core_cash + reserved + core_value + swing_cash + swing_value + parked_value
        equity[day] = total
        swing_equity[day] = swing_cash + swing_value + parked_value
        invested[day] = 1 - (core_cash + reserved + swing_cash) / total if total else Decimal(0)
        peak = max(peak, total)
        # Parked VOO is spendable: a swing buy sells it at the next open.
        spendable = swing_cash + parked_value
        sleeve = swing_equity[day]
        index_day = index
        # Parked VOO's own move today: its value change less the cash parked or unparked.
        parked_pnl = parked_value - parked_before - park_flow if parked_out else Decimal(0)
        if swing_before is not None and swing_before > 0:
            index = index * max(Decimal(0), sleeve - new_swing - parked_pnl) / swing_before
        swing_before, new_swing = sleeve, Decimal(0)
        parked_before, park_flow = parked_value, Decimal(0)
        index_by_day[day] = index
        index_peak = max(index_peak, index)
        if sleeve_breakers:
            # Each anchor is the sleeve now scaled by the index then vs now.
            peak_at, day_at, week_at = (
                sleeve * at / index if index > 0 else sleeve
                for at in (index_peak, index_day, index_week)
            )
            portfolio = _portfolio(
                day,
                spendable,
                swing,
                sleeve,
                peak_at,
                day_at,
                week_at,
                trades,
            ).model_copy(update={"risk_equity": total})
            day_limits = replace(
                limits,
                max_position_pct=SWING_POSITION_CAP,
                cash_reserve_pct=Decimal(0) if sweep else SWING_RESERVE,
            )
        else:
            portfolio = _portfolio(
                day, spendable, swing, total, peak, day_start, week_start, trades
            )
            ratio = sleeve / total
            day_limits = replace(
                limits,
                max_position_pct=SWING_POSITION_CAP * ratio,
                cash_reserve_pct=Decimal(0) if sweep else SWING_RESERVE * ratio,
            )
        day_start = total

        # 4. Swing decision at the close, through the live gate.
        swing_today = {s: b for s, b in today.items() if s in swing_symbols}
        proposals = _proposals(swing_today, day, score_at, TREND_EDGE, swing, pending)
        redis.now = datetime.combine(day, DECISION_TIME, UTC)
        decision = evaluate(
            proposals,
            portfolio,
            {s: _window(candles[s], bar_days[s], day) for s in swing_today},
            {s: b.close for s, b in swing_today.items()},
            MarketEnvironment(),
            kill_switch,
            day_limits,
            now=redis.now,
        )
        for order in decision.approved:
            pending[order.symbol] = (order, today[order.symbol].close)
        for rejection in decision.rejected:
            reason = rejection.reason.value
            rejections[reason] = rejections.get(reason, 0) + 1
        halted = redis.get(KILL_SWITCH_KEY) is not None
        if halted and halted_on is None:
            halted_on = day

        # 5. Core decision at the close: stops, re-entry, then deposit buys.
        for s in [s for s in CORE_TARGETS if s in today]:
            close, line, h = today[s].close, last_line.get(s), core.get(s)
            if s in sold:
                upto = bisect_right(bar_days[s], day)
                since = max(0, bisect_left(bar_days[s], sold[s]) - TREND_DAYS)
                sold_at = datetime.combine(sold[s], time(0), UTC)
                if live.crossed_back(candles[s][since:upto], sold_at):
                    del sold[s]
            if h is not None and s not in core_sells:
                h.high_close = max(h.high_close, close)
                if stop.breached(close, h, line):
                    core_sells.add(s)
        if halted and kill_scope == "everything":
            continue
        eligible = {
            s
            for s in CORE_TARGETS
            if s in last_close
            and s not in sold
            and s not in core_sells
            and not (stop.below_trend and not (s in last_line and last_close[s] > last_line[s]))
        }
        values = {
            s: (core[s].quantity * last_close[s] if s in core else Decimal(0))
            + pending_core.get(s, (Decimal(0),))[0]
            for s in CORE_TARGETS
            if s in last_close
        }
        venue_cash = {"stock": stock_claim + swing_cash, "crypto": crypto_cash}
        for s, dollars in core_buys(stock_claim, venue_cash, values, eligible, total).items():
            atr = wilder_atr(_window(candles[s], bar_days[s], day)) if stop.atr else Decimal(0)
            pending_core[s] = (pending_core.get(s, (Decimal(0),))[0] + dollars, atr)
            if s in CORE_CRYPTO:
                crypto_cash -= dollars
            else:
                stock_claim -= dollars

    def series(values: dict) -> pd.Series:
        """Day -> value as an ascending, day-indexed float series."""
        return pd.Series({pd.Timestamp(d): float(v) for d, v in values.items()}).sort_index()

    return BookResult(
        series(equity),
        series(swing_equity),
        deposits,
        trades,
        core_exits,
        unaffordable,
        halted_on,
        rejections,
        series(invested),
        sweep_trades,
    )


def summarize_book(result: BookResult) -> dict:
    """Headline figures for one simulation: money-weighted returns, drawdown and counters."""
    end = result.equity.index[-1].date()
    final = float(result.equity.iloc[-1])
    flows = [(d, float(a)) for d, a in sorted(result.deposits.items())]
    swing_flows = [(d, a * float(1 - CORE_SHARE)) for d, a in flows]
    deposited = sum(a for _, a in flows)
    wins = [t for t in result.swing_trades if t.pnl > 0]
    deposit_series = pd.Series({pd.Timestamp(d): a for d, a in flows})
    return {
        "deposited": round(deposited, 2),
        "final": round(final, 2),
        "profit": round(final - deposited, 2),
        "money_weighted": money_weighted_return(flows, end, final),
        "max_drawdown": time_weighted_drawdown(result.equity, deposit_series),
        "swing_money_weighted": money_weighted_return(
            swing_flows, end, float(result.swing_equity.iloc[-1])
        ),
        "swing_trades": len(result.swing_trades),
        "swing_win_rate": round(len(wins) / len(result.swing_trades), 4)
        if result.swing_trades
        else 0.0,
        "unaffordable": result.unaffordable,
        "core_exits": result.core_exits,
        "kill_switch": result.halted_on.isoformat() if result.halted_on else None,
        "invested": round(float(result.invested.mean()), 4)
        if result.invested is not None
        else None,
        "sweep_trades": result.sweep_trades,
    }


def benchmark(history: dict[str, list[Candle]], deposits: dict[date, Decimal], end: date) -> dict:
    """Every deposit into VOO, never sold: the reference for the whole book and the swing sleeve.

    The swing sleeve's share of each deposit in VOO would earn the same money-weighted rate.
    """
    amounts = {d: float(a) for d, a in deposits.items()}
    curve = deposit_and_hold(history["VOO"], amounts, end, SLIPPAGE)
    final = float(curve.iloc[-1])
    flows = sorted(amounts.items())
    deposited = sum(amounts.values())
    return {
        "deposited": round(deposited, 2),
        "final": round(final, 2),
        "profit": round(final - deposited, 2),
        "money_weighted": money_weighted_return(flows, end, final),
        "max_drawdown": time_weighted_drawdown(
            curve, pd.Series({pd.Timestamp(d): a for d, a in flows})
        ),
    }


def first_day(history: dict[str, list[Candle]]) -> date:
    """The first day every core and swing lookback is available; ValueError if a symbol is short."""
    ready = []
    for s, cs in history.items():
        need = TREND_DAYS if s in CORE_TARGETS else SWING_WARMUP
        ordered = sorted(c.ts.date() for c in cs)
        if len(ordered) < need:
            raise ValueError(f"{s}: {len(ordered)} bars, the study needs {need}")
        ready.append(ordered[need - 1])
    return max(ready) + timedelta(days=1)


def study(history: dict[str, list[Candle]]) -> dict:
    """Run every core stop x kill-switch scope on both halves, ranked by tune-half return.

    Each half starts from its own first deposit, so the halves are independent accounts.
    """
    start = first_day(history)
    end = max(c.ts.date() for cs in history.values() for c in cs)
    mid = start + (end - start) / 2
    halves = {"tune": (start, mid - timedelta(days=1)), "validate": (mid, end)}
    limits = backtest_limits()
    runs, benchmarks = [], {}
    for (name, stop), scope in product(CORE_STOPS.items(), KILL_SCOPES):
        run = {"core_stop": name, "kill_switch_scope": scope}
        for half, (a, b) in halves.items():
            clipped = {s: [c for c in cs if c.ts.date() <= b] for s, cs in history.items()}
            result = simulate_book(clipped, a, b, stop, scope, limits)
            run[half] = {**summarize_book(result), "rejections": result.rejections}
            benchmarks.setdefault(half, benchmark(clipped, result.deposits, b))
        runs.append(run)
    runs.sort(key=lambda r: r["tune"]["money_weighted"], reverse=True)
    return {
        **{h: (a.isoformat(), b.isoformat()) for h, (a, b) in halves.items()},
        "benchmark": benchmarks,
        "runs": runs,
    }


SWEEP_CONFIGS = {
    "D-039 (account breakers, 20% reserve)": {"sleeve_breakers": False, "sweep": False},
    "D-042 (sleeve breakers, 20% reserve)": {"sleeve_breakers": True, "sweep": False},
    "D-043 (sleeve breakers, VOO sweep)": {"sleeve_breakers": True, "sweep": True},
    "D-043 (b) (breakers leave parked VOO out)": {
        "sleeve_breakers": True,
        "sweep": True,
        "parked_out": True,
    },
}


def sweep_study(history: dict[str, list[Candle]]) -> dict:
    """Run the chosen plan (25% trail, swing-only kill switch) under each SWEEP_CONFIGS entry."""
    start = first_day(history)
    end = max(c.ts.date() for cs in history.values() for c in cs)
    mid = start + (end - start) / 2
    halves = {"tune": (start, mid - timedelta(days=1)), "validate": (mid, end)}
    runs, benchmarks = [], {}
    for name, options in SWEEP_CONFIGS.items():
        run: dict = {"config": name}
        for half, (a, b) in halves.items():
            clipped = {s: [c for c in cs if c.ts.date() <= b] for s, cs in history.items()}
            result = simulate_book(
                clipped,
                a,
                b,
                CORE_STOPS["trail 25%"],
                "swing only",
                backtest_limits(),
                **options,
            )
            run[half] = {**summarize_book(result), "rejections": result.rejections}
            benchmarks.setdefault(half, benchmark(clipped, result.deposits, b))
        runs.append(run)
    return {
        **{h: (a.isoformat(), b.isoformat()) for h, (a, b) in halves.items()},
        "benchmark": benchmarks,
        "runs": runs,
    }


def format_sweep(report: dict) -> str:
    """Render sweep_study() as a plain-text with/without table."""
    lines = [
        "25% core trail, kill switch on the swing sleeve; "
        f"tune {report['tune'][0]} -> {report['tune'][1]}, "
        f"validate {report['validate'][0]} -> {report['validate'][1]}",
        f"{'':<44}{'half':<10}{'invested':>9}{'MWR':>7}{'maxDD':>7}{'final':>10}"
        f"{'swing MWR':>10}{'swing trades':>13}{'sweep trades':>13}  kill switch",
    ]
    for r in report["runs"]:
        for half in ("tune", "validate"):
            m = r[half]
            lines.append(
                f"{r['config']:<44}{half:<10}{m['invested']:>9.1%}{m['money_weighted']:>7.1%}"
                f"{m['max_drawdown']:>7.1%}{m['final']:>10,.0f}{m['swing_money_weighted']:>10.1%}"
                f"{m['swing_trades']:>13}{m['sweep_trades']:>13}  {m['kill_switch'] or '-'}"
            )
    for half, m in report["benchmark"].items():
        lines.append(
            f"every deposit into VOO, {half}: final {m['final']:,.0f} of {m['deposited']:,.0f}, "
            f"MWR {m['money_weighted']:.1%}, max drawdown {m['max_drawdown']:.1%}"
        )
    lines.append("MWR = money-weighted yearly return; maxDD = time-weighted. Past results only.")
    return "\n".join(lines)


def format_table(report: dict) -> str:
    """Render study()'s runs and the VOO reference as a plain-text table."""
    tune, val = report["tune"], report["validate"]
    lines = [
        f"tune {tune[0]} -> {tune[1]}   validate {val[0]} -> {val[1]}   "
        f"($1,000 + $500/14 days, 70% core / 30% swing; {SLIPPAGE:.1%} slippage, "
        "crypto +0.6% fee)",
        f"{'core stop':<14}{'kill switch':<12}{'tune MWR':>9}{'maxDD':>7}  |"
        f"{'deposited':>10}{'final':>10}{'MWR':>7}{'maxDD':>7}{'swing MWR':>10}"
        f"{'trades':>7}{'core exits':>11}{'no share':>9}  kill switch",
    ]
    for r in report["runs"]:
        t, v = r["tune"], r["validate"]
        lines.append(
            f"{r['core_stop']:<14}{r['kill_switch_scope']:<12}{t['money_weighted']:>9.1%}"
            f"{t['max_drawdown']:>7.1%}  |{v['deposited']:>10,.0f}{v['final']:>10,.0f}"
            f"{v['money_weighted']:>7.1%}{v['max_drawdown']:>7.1%}"
            f"{v['swing_money_weighted']:>10.1%}{v['swing_trades']:>7}{v['core_exits']:>11}"
            f"{v['unaffordable']:>9}  tune {t['kill_switch'] or '-'} / validate "
            f"{v['kill_switch'] or '-'}"
        )
    for half, m in report["benchmark"].items():
        lines.append(
            f"every deposit into VOO, {half}: final {m['final']:,.0f} of {m['deposited']:,.0f}, "
            f"MWR {m['money_weighted']:.1%}, max drawdown {m['max_drawdown']:.1%}"
        )
    lines.append("MWR = money-weighted yearly return; maxDD = time-weighted. Past results only.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: fetch history, run the study, write the JSON report and print the table."""
    parser = argparse.ArgumentParser(prog="trading_agent.backtest.core_swing")
    parser.add_argument("--sweep", action="store_true", help="D-043: with vs without the sweep")
    args = parser.parse_args(argv)
    configure_logging(logging.CRITICAL)  # see backtest.run: the report carries the events
    settings = load_settings()
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        raise SystemExit("the study needs ALPACA_API_KEY and ALPACA_SECRET_KEY (history source)")
    now = datetime.now(UTC)
    last_complete = datetime.combine(now.date(), time(0), UTC) - timedelta(seconds=1)
    client = AlpacaDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    history = fetch_history(client, CORE_STOCKS + SWING_SYMBOLS, CORE_CRYPTO, last_complete)
    report = sweep_study(history) if args.sweep else study(history)
    OUT_DIR.mkdir(exist_ok=True)
    name = "core-swing-sweep" if args.sweep else "core-swing"
    path = OUT_DIR / f"{name}-{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(format_sweep(report) if args.sweep else format_table(report))
    print(f"\nfull report: {path}")


if __name__ == "__main__":
    main()
