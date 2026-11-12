"""Stop-loss and signal study (D-031): which exits and entries survive costs?

`python -m trading_agent.backtest.study` runs a grid of entry signal (model, trend, always)
x ATR multiple x trail rule x loss-streak pause x position cap through the live gate at
paper costs. Configurations are ranked on the tune half and reported on the validate half.
"""

import argparse
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from itertools import product

import pandas as pd

from trading_agent.backtest.engine import Params, StopRule, backtest_limits, simulate
from trading_agent.backtest.metrics import buy_and_hold, exposure_matched, summarize
from trading_agent.backtest.run import OUT_DIR, model_scores
from trading_agent.brokers.paper import SLIPPAGE_PCT
from trading_agent.config import load_settings
from trading_agent.data.models import Candle
from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.log import configure_logging
from trading_agent.ml.train import fetch_history

# Crypto also pays its taker fee (engine.COINBASE_TAKER_FEE).
SLIPPAGE = SLIPPAGE_PCT
STARTING_CASH = Decimal(100_000)
MODEL_EDGE = 0.05
# RISK_RULES ranges: ATR multiple 1.5-3.0; loss-streak pause after 5-7 losses.
ATR_MULTIPLES = (Decimal("2.0"), Decimal("3.0"))
LOSS_STREAKS = (5, 7)
# RISK_RULES range: 5-10% per position.
POSITION_CAPS = (Decimal("0.05"), Decimal("0.10"))
PRE_D032 = "trail 3% intraday (pre-D-032)"
STOP_RULES = {
    PRE_D032: StopRule(trail_pct=Decimal("0.03")),
    "trail 3% on close": StopRule(trail_pct=Decimal("0.03"), trail_on_close=True),
    "trail 8% intraday": StopRule(trail_pct=Decimal("0.08")),
    "trail 8% on close": StopRule(trail_pct=Decimal("0.08"), trail_on_close=True),
    "trail 3xATR on close": StopRule(trail_atr=Decimal(3), trail_on_close=True),
    "ATR stop only": StopRule(),
}


def trend_scores(history: dict[str, list[Candle]], days: pd.Series) -> pd.DataFrame:
    """A trend signal in engine.simulate's scores format, base rate 0.5.

    Score 1.0 when the close is above its 50-day average and the 20-day is above the 50-day,
    else 0.0; each value reads bars up to its day only.
    """
    frames = []
    window = pd.DatetimeIndex(days)
    for symbol, candles in history.items():
        closes = pd.Series(
            {pd.Timestamp(c.ts.date()): float(c.close) for c in candles}
        ).sort_index()
        sma20, sma50 = closes.rolling(20).mean(), closes.rolling(50).mean()
        up = ((closes > sma50) & (sma20 > sma50)).astype(float).where(sma50.notna())
        rows = up.dropna()
        rows = rows[rows.index.tz_localize(UTC).isin(window)]
        frames.append(
            pd.DataFrame(
                {
                    "ts": pd.to_datetime(rows.index).tz_localize(UTC),
                    "symbol": symbol,
                    "score": rows.to_numpy(),
                    "base_rate": 0.5,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def always_scores(model: pd.DataFrame) -> pd.DataFrame:
    """A buy-whenever-flat signal on the model's scored days; it never exits on signal."""
    return model.assign(score=1.0, base_rate=0.5)


def study(history: dict[str, list[Candle]]) -> dict:
    """Run the grid on the tune and validate halves, ranked by tune-half Sharpe."""
    model = model_scores(history)
    model["ts"] = pd.to_datetime(model["ts"], utc=True)
    days = pd.Series(sorted(model["ts"].unique()))
    signals = {
        "model": (model, MODEL_EDGE),
        "trend": (trend_scores(history, days), 0.25),
        "always": (always_scores(model), 0.25),
    }
    mid = days.iloc[len(days) // 2]
    last = max(c.ts.date() for cs in history.values() for c in cs)
    halves = {
        "tune": (days.iloc[0], mid - pd.Timedelta(days=1)),
        "validate": (mid, pd.Timestamp(last, tz=UTC)),
    }
    cash = float(STARTING_CASH)
    benchmarks = {
        name: buy_and_hold(history, start.date(), end.date(), SLIPPAGE, STARTING_CASH)
        for name, (start, end) in halves.items()
    }

    runs = []
    for (signal, (scores, edge)), atr, streak, cap, (stop_name, rule) in product(
        signals.items(), ATR_MULTIPLES, LOSS_STREAKS, POSITION_CAPS, STOP_RULES.items()
    ):
        limits = replace(
            backtest_limits(),
            atr_multiplier=atr,
            max_consecutive_losses=streak,
            max_position_pct=cap,
        )
        params = Params(edge, SLIPPAGE, STARTING_CASH, stops=rule)
        result = {
            "signal": signal,
            "atr_multiple": str(atr),
            "loss_streak": streak,
            "position_cap": str(cap),
            "stops": stop_name,
        }
        for half, (start, end) in halves.items():
            window = scores[(scores["ts"] >= start) & (scores["ts"] <= end)]
            clipped = {s: [c for c in cs if c.ts <= end] for s, cs in history.items()}
            sim = simulate(clipped, window, params, limits)
            exposure = float(sim.exposure.mean())
            result[half] = {
                **summarize(sim.equity, cash, sim.trades),
                "mean_exposure": round(exposure, 4),
                "bh_at_exposure": summarize(
                    exposure_matched(benchmarks[half], cash, exposure), cash
                )["total_return"],
                "halted": sim.halted_on is not None,
            }
        runs.append(result)
    runs.sort(key=lambda r: r["tune"]["sharpe"], reverse=True)
    return {
        **{h: (s.date().isoformat(), e.date().isoformat()) for h, (s, e) in halves.items()},
        "benchmark": {h: summarize(b, cash) for h, b in benchmarks.items()},
        "runs": runs,
    }


def format_table(report: dict, top: int = 12) -> str:
    """Render the `top` configurations by tune-half Sharpe and the pre-D-032 baseline."""
    tune, val = report["tune"], report["validate"]
    header = (
        f"{'#':>3} {'signal':<7}{'ATRx':>5}{'streak':>7}{'cap':>5}  {'stops':<30}"
        f"{'tune ret':>9}{'Sharpe':>7}{'B&H@exp':>8}  |{'valid ret':>10}{'Sharpe':>7}"
        f"{'maxDD':>7}{'trades':>7}{'exp':>5}{'B&H@exp':>8}"
    )
    lines = [
        f"tune {tune[0]} -> {tune[1]}   validate {val[0]} -> {val[1]}   "
        f"(paper costs: {SLIPPAGE:.1%} slippage, crypto +0.6% fee)",
        header,
    ]

    def find(atr: str, cap: str, stops: str) -> int:
        """Index in the ranked runs of the model-signal run with these settings (streak 5)."""
        return next(
            i
            for i, r in enumerate(report["runs"])
            if (r["signal"], r["atr_multiple"], r["loss_streak"], r["position_cap"], r["stops"])
            == ("model", atr, 5, cap, stops)
        )

    # The ML signal at each rule set - the live decision-maker is the LLM, which no
    # backtest can replay, so these rows show the limits, not the live strategy.
    marked = {
        find("2.0", "0.05", PRE_D032): "  <- pre-D-032 limits",
        find("3.0", "0.10", "ATR stop only"): "  <- D-032 limits",
    }
    for i in [*range(min(top, len(report["runs"]))), *marked]:
        r = report["runs"][i]
        t, v = r["tune"], r["validate"]
        lines.append(
            f"{i + 1:>3} {r['signal']:<7}{r['atr_multiple']:>5}{r['loss_streak']:>7}"
            f"{float(r['position_cap']):>5.0%}  "
            f"{r['stops']:<30}{t['total_return']:>9.1%}{t['sharpe']:>7.2f}"
            f"{t['bh_at_exposure']:>8.1%}  |{v['total_return']:>10.1%}{v['sharpe']:>7.2f}"
            f"{v['max_drawdown']:>7.1%}{v['trades']:>7}{v['mean_exposure']:>5.0%}"
            f"{v['bh_at_exposure']:>8.1%}" + marked.get(i, "")
        )
    for half, m in report["benchmark"].items():
        lines.append(
            f"buy-and-hold fully invested, {half}: {m['total_return']:.1%}, "
            f"Sharpe {m['sharpe']:.2f}, max drawdown {m['max_drawdown']:.1%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: fetch history, run the study, write the JSON report and print the table."""
    argparse.ArgumentParser(prog="trading_agent.backtest.study").parse_args(argv)
    configure_logging(logging.CRITICAL)  # see backtest.run: the report carries the events
    settings = load_settings()
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        raise SystemExit("the study needs ALPACA_API_KEY and ALPACA_SECRET_KEY (history source)")
    now = datetime.now(UTC)
    last_complete = datetime.combine(now.date(), time(0), UTC) - timedelta(seconds=1)
    client = AlpacaDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    history = fetch_history(client, settings.stock_symbols, settings.crypto_symbols, last_complete)
    report = study(history)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"study-{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(format_table(report))
    print(f"\nfull report: {path}")


if __name__ == "__main__":
    main()
