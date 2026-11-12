"""Train the directional model offline: python -m trading_agent.ml.train

Fetches years of daily history from Alpaca, measures skill with a purged
expanding-window walk-forward, fits the final model on every row and saves a new
version. The cycle serves it only if model.is_servable passes. Walk-forward rather
than k-fold so no fold ever trains on information from its own test period.
"""

import argparse
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import xgboost as xgb

from trading_agent.config import Settings, load_settings
from trading_agent.data.models import Candle
from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.log import configure_logging, get_logger
from trading_agent.ml import model
from trading_agent.ml.dataset import FEATURES, HORIZON_BARS, labelled_rows

log = get_logger(__name__)

# Alpaca's IEX daily history starts mid-2020, crypto in 2021; ask from before both.
HISTORY_START = datetime(2016, 1, 1, tzinfo=UTC)
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "seed": 0,
}
NUM_ROUNDS = 200
FOLDS = 5
# The first half of the timeline is the minimum training set before any test block.
MIN_TRAIN_FRACTION = 0.5


def fetch_history(
    client: AlpacaDataClient, stocks: list[str], crypto: list[str], end: datetime
) -> dict[str, list[Candle]]:
    """Daily bars up to end for every watchlist symbol; symbols with none are left out.

    Pass an end before today's 00:00 UTC so the forming bar is never trained on. A
    failed fetch raises rather than training on a silently incomplete set.
    """
    history: dict[str, list[Candle]] = {}
    for symbol in stocks:
        history[symbol] = client.get_daily_candles(symbol, HISTORY_START, end)
    for symbol in crypto:
        history[symbol] = client.get_crypto_daily_candles(symbol, HISTORY_START, end)
    missing = sorted(s for s, c in history.items() if not c)
    if missing:
        log.warning("training_symbols_without_history", symbols=missing)
    return {s: c for s, c in history.items() if c}


def build_dataset(history: dict[str, list[Candle]]) -> pd.DataFrame:
    """Every symbol's labelled rows in one frame, sorted by (ts, symbol); ValueError if none."""
    if not history:
        raise ValueError("no training history for any watchlist symbol")
    frames = [labelled_rows(candles, symbol) for symbol, candles in sorted(history.items())]
    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["ts", "symbol"], ignore_index=True)


def auc(labels: pd.Series, scores: pd.Series) -> float:
    """ROC AUC by the rank-sum identity; 0.5 when only one class is present."""
    labels = labels.reset_index(drop=True)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = scores.reset_index(drop=True).rank(method="average")
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def fit(rows: pd.DataFrame) -> xgb.Booster:
    """Train a booster on rows with FEATURES and "label" (deterministic: fixed seed)."""
    matrix = xgb.DMatrix(rows[list(FEATURES)], label=rows["label"], feature_names=list(FEATURES))
    return xgb.train(PARAMS, matrix, num_boost_round=NUM_ROUNDS)


def walk_forward_scores(data: pd.DataFrame) -> pd.DataFrame:
    """The test rows of a purged walk-forward, with "score" (out of sample) and "fold".

    The dates after the first MIN_TRAIN_FRACTION are cut into FOLDS consecutive test
    blocks, each scored by a model trained only on rows whose label_end falls before
    the block. The backtest trades on these scores. Too little history raises ValueError.
    """
    dates = pd.Series(data["ts"].unique()).sort_values(ignore_index=True)
    test_dates = dates[int(len(dates) * MIN_TRAIN_FRACTION) :].reset_index(drop=True)
    if len(test_dates) < FOLDS:
        raise ValueError(f"only {len(dates)} distinct dates - too little history")
    block = len(test_dates) // FOLDS
    scored: list[pd.DataFrame] = []
    for k in range(FOLDS):
        start = test_dates[k * block]
        end = test_dates[(k + 1) * block] if k < FOLDS - 1 else None
        train_rows = data[data["label_end"] < start]
        if train_rows.empty:
            raise ValueError(f"fold {k} has no training rows before {start} - too little history")
        in_block = data["ts"] >= start if end is None else data["ts"].between(start, end, "left")
        test_rows = data[in_block].copy()
        test_rows["score"] = fit(train_rows).predict(
            xgb.DMatrix(test_rows[list(FEATURES)], feature_names=list(FEATURES))
        )
        test_rows["fold"] = k
        scored.append(test_rows)
    return pd.concat(scored, ignore_index=True)


def walk_forward(data: pd.DataFrame) -> dict:
    """Pooled out-of-sample "auc", "accuracy", "base_rate", "n_test", plus "fold_auc" per block."""
    oos = walk_forward_scores(data)
    return {
        "auc": round(auc(oos["label"], oos["score"]), 4),
        "accuracy": round(float(((oos["score"] >= 0.5) == (oos["label"] == 1)).mean()), 4),
        "base_rate": round(float(oos["label"].mean()), 4),
        "n_test": len(oos),
        "fold_auc": [
            round(auc(g["label"], g["score"]), 4) for _, g in oos.groupby("fold", sort=True)
        ],
    }


def train(history: dict[str, list[Candle]], root: Path, now: datetime) -> dict:
    """Evaluate, fit on everything and save a version named after now; returns its meta."""
    data = build_dataset(history)
    metrics = walk_forward(data)
    booster = fit(data)
    meta = {
        "version": now.strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": now.isoformat(),
        "features": list(FEATURES),
        "horizon_bars": HORIZON_BARS,
        "params": PARAMS,
        "num_rounds": NUM_ROUNDS,
        "xgboost_version": xgb.__version__,
        "symbols": sorted(history),
        "n_rows": len(data),
        "first_ts": data["ts"].min().isoformat(),
        "last_ts": data["ts"].max().isoformat(),
        "walk_forward": metrics,
    }
    path = model.save(booster, meta, root)
    log.info(
        "model_trained",
        version=meta["version"],
        path=str(path),
        n_rows=meta["n_rows"],
        walk_forward=metrics,
        served=model.is_servable(metrics),
        min_oos_auc=model.MIN_OOS_AUC,
    )
    return meta


def train_from_alpaca(settings: Settings, root: Path) -> dict:
    """Fetch the watchlists' history from Alpaca and train a version (CLI and weekly retrain)."""
    now = datetime.now(UTC)
    last_complete = datetime.combine(now.date(), time(0), UTC) - timedelta(seconds=1)
    client = AlpacaDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    history = fetch_history(client, settings.stock_symbols, settings.crypto_symbols, last_complete)
    return train(history, root, now)


def main(argv: list[str] | None = None) -> None:
    """CLI: train a version; exits non-zero when the Alpaca keys are missing."""
    argparse.ArgumentParser(prog="trading_agent.ml.train").parse_args(argv)

    configure_logging()
    settings = load_settings()
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        raise SystemExit("training needs ALPACA_API_KEY and ALPACA_SECRET_KEY (history source)")
    train_from_alpaca(settings, model.MODEL_DIR)


if __name__ == "__main__":
    main()
