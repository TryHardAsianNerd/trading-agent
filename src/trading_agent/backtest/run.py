"""Backtest CLI: `python -m trading_agent.backtest.run [--edges ...] [--slippages ...]`.

Scores the watchlists' history with the purged walk-forward and replays every edge x
slippage configuration through the live risk gate, next to buy-and-hold. Writes the full
result to backtests/. Past results only; nothing here is a forecast.
"""

import argparse
import json
import logging
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from trading_agent.backtest.engine import Params, backtest_limits, simulate
from trading_agent.backtest.metrics import buy_and_hold, exposure_matched, summarize
from trading_agent.config import load_settings
from trading_agent.data.models import Candle
from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.log import configure_logging
from trading_agent.ml.train import build_dataset, fetch_history, walk_forward_scores

OUT_DIR = Path("backtests")
DEFAULT_EDGES = (0.0, 0.02, 0.05)
# Both ends of RISK_RULES' 0.1-0.5% slippage range.
DEFAULT_SLIPPAGES = (Decimal("0.001"), Decimal("0.005"))


def model_scores(history: dict[str, list[Candle]]) -> pd.DataFrame:
    """The model's out-of-sample scores, each with its fold's training base rate."""
    data = build_dataset(history)
    scores = walk_forward_scores(data)
    fold_starts = scores.groupby("fold")["ts"].min()
    fold_rates = {
        k: float(data.loc[data["label_end"] < t, "label"].mean()) for k, t in fold_starts.items()
    }
    scores["base_rate"] = scores["fold"].map(fold_rates)
    return scores


def run(
    history: dict[str, list[Candle]],
    edges: list[float],
    slippages: list[Decimal],
    starting_cash: Decimal = Decimal(100_000),
) -> dict:
    """Score the history out of sample and backtest every edge x slippage configuration.

    Each run carries buy-and-hold matched to its exposure; each slippage gets a full benchmark.
    """
    scores = model_scores(history)
    start = scores["ts"].min().date()
    end = max(c.ts.date() for cs in history.values() for c in cs)
    limits = backtest_limits()
    cash = float(starting_cash)

    runs, benchmark = [], []
    for slippage in slippages:
        bh = buy_and_hold(history, start, end, slippage, starting_cash)
        benchmark.append({"slippage": str(slippage), "metrics": summarize(bh, cash)})
        for edge in edges:
            result = simulate(history, scores, Params(edge, slippage, starting_cash), limits)
            exposure = float(result.exposure.mean())
            metrics = summarize(result.equity, cash, result.trades)
            runs.append(
                {
                    "edge": edge,
                    "slippage": str(slippage),
                    "metrics": {**metrics, "mean_exposure": round(exposure, 4)},
                    "matched_benchmark": summarize(exposure_matched(bh, cash, exposure), cash),
                    "halted_on": result.halted_on.isoformat() if result.halted_on else None,
                    "rejections": result.rejections,
                }
            )
    return {
        "window": {"start": start.isoformat(), "end": end.isoformat(), "symbols": sorted(history)},
        "fold_base_rates": [
            round(float(r), 4) for r in scores.groupby("fold")["base_rate"].first()
        ],
        "runs": runs,
        "benchmark": benchmark,
    }


def format_table(report: dict) -> str:
    """Render run()'s report as a fixed-width table."""
    w = report["window"]
    rates = ", ".join(f"{r:.3f}" for r in report["fold_base_rates"])
    lines = [
        f"Backtest {w['start']} -> {w['end']}  symbols {', '.join(w['symbols'])}",
        f"fold base rates {rates}   B&H@exp = buy-and-hold at the run's average exposure",
        f"{'config':<22}{'return':>8}{'Sharpe':>7}{'maxDD':>7}{'trades':>7}{'win%':>6}"
        f"{'PF':>6}{'exp':>6}{'B&H@exp':>9}{'Sharpe':>7}  halted",
    ]
    for r in report["runs"]:
        m, b = r["metrics"], r["matched_benchmark"]
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] is not None else "-"
        lines.append(
            f"{'edge ' + str(r['edge']) + ' slip ' + r['slippage']:<22}"
            f"{m['total_return']:>8.1%}{m['sharpe']:>7.2f}{m['max_drawdown']:>7.1%}"
            f"{m['trades']:>7}{m['win_rate']:>6.0%}{pf:>6}{m['mean_exposure']:>6.0%}"
            f"{b['total_return']:>9.1%}{b['sharpe']:>7.2f}  {r['halted_on'] or '-'}"
        )
    for b in report["benchmark"]:
        m = b["metrics"]
        lines.append(
            f"{'buy&hold slip ' + b['slippage']:<22}"
            f"{m['total_return']:>8.1%}{m['sharpe']:>7.2f}{m['max_drawdown']:>7.1%}"
            f"{'':>19}{'100%':>6}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI: fetch history, run the backtest, write the JSON report and print the table."""
    parser = argparse.ArgumentParser(prog="trading_agent.backtest.run")
    parser.add_argument("--edges", default=",".join(map(str, DEFAULT_EDGES)))
    parser.add_argument("--slippages", default=",".join(map(str, DEFAULT_SLIPPAGES)))
    args = parser.parse_args(argv)

    # The gate logs every decision and breaker trip; the report carries them, so only
    # CRITICAL reaches the console.
    configure_logging(logging.CRITICAL)
    settings = load_settings()
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        raise SystemExit("backtesting needs ALPACA_API_KEY and ALPACA_SECRET_KEY (history source)")
    now = datetime.now(UTC)
    last_complete = datetime.combine(now.date(), time(0), UTC) - timedelta(seconds=1)
    client = AlpacaDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    history = fetch_history(client, settings.stock_symbols, settings.crypto_symbols, last_complete)

    report = run(
        history,
        [float(e) for e in args.edges.split(",")],
        [Decimal(s) for s in args.slippages.split(",")],
    )
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(format_table(report))
    print(f"\nfull report: {path}")


if __name__ == "__main__":
    main()
