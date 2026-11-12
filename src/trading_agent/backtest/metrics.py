"""Backtest performance metrics and the buy-and-hold benchmarks.

Equity curves are calendar-daily (crypto trades every day), so Sharpe is annualized with
sqrt(365) and a zero risk-free rate. Figures describe the past only; nothing here predicts
future returns.
"""

import math
from datetime import date, timedelta
from decimal import Decimal

import pandas as pd

from trading_agent.backtest.engine import COINBASE_TAKER_FEE, ClosedTrade
from trading_agent.data.models import Candle

DAYS_PER_YEAR = 365


def summarize(
    equity: pd.Series, starting_cash: float, trades: list[ClosedTrade] | None = None
) -> dict:
    """Headline metrics for an equity curve and its trades, if any, rounded to 4 dp.

    Returns are measured from the starting cash on the eve of the first day, so the first
    day's fills count. profit_factor is None when no trade lost.
    """
    eve = pd.Series([float(starting_cash)], index=[equity.index[0] - pd.Timedelta(days=1)])
    equity = pd.concat([eve, equity])
    returns = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days, 1) / DAYS_PER_YEAR
    growth = equity.iloc[-1] / equity.iloc[0]
    std = returns.std()
    out = {
        "total_return": growth - 1,
        "cagr": growth ** (1 / years) - 1,
        "sharpe": returns.mean() / std * math.sqrt(DAYS_PER_YEAR) if std > 0 else 0.0,
        "max_drawdown": float((1 - equity / equity.cummax()).max()),
    }
    if trades is not None:
        wins = [float(t.pnl) for t in trades if t.pnl > 0]
        losses = [-float(t.pnl) for t in trades if t.pnl < 0]
        out["trades"] = len(trades)
        out["win_rate"] = len(wins) / len(trades) if trades else 0.0
        out["profit_factor"] = sum(wins) / sum(losses) if losses else None
    return {k: round(float(v), 4) if isinstance(v, float) else v for k, v in out.items()}


def buy_and_hold(
    history: dict[str, list[Candle]],
    start: date,
    end: date,
    slippage: Decimal,
    starting_cash: Decimal,
    crypto_fee: Decimal = COINBASE_TAKER_FEE,
) -> pd.Series:
    """Equal-weight buy-and-hold of every symbol, as end-of-day equity by calendar day.

    Each symbol is bought at its first open on or after `start`, after costs, and marked at
    its latest close.
    """
    per_symbol = float(starting_cash) / len(history)
    curves = []
    for candles in history.values():
        closes = pd.Series(
            {
                pd.Timestamp(c.ts.date()): float(c.close)
                for c in candles
                if start <= c.ts.date() <= end
            }
        ).sort_index()
        first = next(c for c in sorted(candles, key=lambda c: c.ts) if c.ts.date() >= start)
        fee = crypto_fee if first.asset_class == "crypto" else Decimal(0)
        units = per_symbol / (float(first.open) * (1 + float(slippage + fee)))
        curves.append(closes * units)
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    # Before a symbol's first bar (a stock on a weekend start) its share is still cash.
    return pd.concat(curves, axis=1).reindex(days).ffill().fillna(per_symbol).sum(axis=1)


def money_weighted_return(flows: list[tuple[date, float]], end: date, final_value: float) -> float:
    """Annualized money-weighted return (XIRR): the one yearly rate every deposit earned.

    Found by bisection on [-99%, +1000%]; a rate outside that comes back pinned to the nearer
    bound, and 0.0 when nothing was deposited.
    """
    if not flows:
        return 0.0

    def surplus(rate: float) -> float:
        """Final value minus every deposit grown at `rate` to `end`."""
        grown = sum(a * (1 + rate) ** ((end - d).days / DAYS_PER_YEAR) for d, a in flows)
        return final_value - grown

    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if surplus(mid) > 0 else (lo, mid)
    return round(mid, 4)


def time_weighted_drawdown(equity: pd.Series, deposits: pd.Series) -> float:
    """Largest peak-to-trough fall of the strategy itself, deposits removed, as a fraction.

    A deposit landing mid-slide would hide the fall, so each day's growth is measured on the
    money already there.
    """
    flows = deposits.reindex(equity.index, fill_value=0.0)
    growth = (equity - flows) / equity.shift(1)
    index = growth.iloc[1:].cumprod()
    if index.empty:
        return 0.0
    index = pd.concat([pd.Series([1.0]), index], ignore_index=True)
    return round(float((1 - index / index.cummax()).max()), 4)


def deposit_and_hold(
    candles: list[Candle],
    deposits: dict[date, float],
    end: date,
    slippage: Decimal,
) -> pd.Series:
    """One stock bought with every deposit and never sold, as end-of-day equity by calendar day.

    A deposit buys at the first open after its day, as the strategy does, so neither side
    gets a head start.
    """
    bars = {c.ts.date(): c for c in candles}
    units, investable, arrived_today, close = 0.0, 0.0, 0.0, None
    out = {}
    day = min(deposits)
    while day <= end:
        investable += arrived_today  # yesterday's deposit may buy from today's open
        arrived_today = deposits.get(day, 0.0)
        bar = bars.get(day)
        if bar is not None:
            if investable:
                units += investable / (float(bar.open) * (1 + float(slippage)))
                investable = 0.0
            close = float(bar.close)
        out[pd.Timestamp(day)] = investable + arrived_today + units * (close or 0.0)
        day += timedelta(days=1)
    return pd.Series(out)


def exposure_matched(benchmark: pd.Series, starting_cash: float, exposure: float) -> pd.Series:
    """The benchmark scaled to a strategy's average invested fraction, the rest in cash.

    The position cap and cash reserve keep the strategy from ever being fully invested.
    """
    return starting_cash * (1 - exposure) + benchmark * exposure
