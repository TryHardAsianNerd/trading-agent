"""Directional-model training: AUC, purged walk-forward, the end-to-end job."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
import structlog
from tests.ml_fakes import candles_from_closes, random_walk

from trading_agent.ml import model, train
from trading_agent.ml.dataset import FEATURES

NOW = datetime(2026, 9, 22, 12, tzinfo=UTC)


def _history(trend_block: int | None, n: int = 700) -> dict:
    return {
        sym: candles_from_closes(random_walk(n, seed, trend_block), symbol=sym)
        for seed, sym in enumerate(["AAA", "BBB", "CCC"])
    }


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),
        ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.0),
        ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5),
        ([0, 1, 1], [0.2, 0.1, 0.9], 0.5),  # one of two positives outranks the negative
        ([1, 1, 1], [0.1, 0.2, 0.3], 0.5),  # one class: no ranking exists
    ],
)
def test_auc_known_values(labels, scores, expected):
    assert train.auc(pd.Series(labels), pd.Series(scores)) == pytest.approx(expected)


def test_walk_forward_trains_only_on_rows_whose_label_ends_before_the_test_block(monkeypatch):
    """The no-lookahead property of the evaluation itself: every training row's
    label window closes strictly before its test block opens, and the test
    blocks tile the second half of the timeline in order without overlap."""
    data = train.build_dataset(_history(None, n=300))
    real_fit = train.fit
    seen: list[pd.DataFrame] = []

    def spy_fit(rows):
        seen.append(rows)
        return real_fit(rows)

    monkeypatch.setattr(train, "fit", spy_fit)
    metrics = train.walk_forward(data)

    assert len(seen) == train.FOLDS
    dates = sorted(data["ts"].unique())
    test_dates = dates[int(len(dates) * train.MIN_TRAIN_FRACTION) :]
    block = len(test_dates) // train.FOLDS
    for k, rows in enumerate(seen):
        assert rows["label_end"].max() < test_dates[k * block]
    # Expanding window: each fold trains on at least as much as the one before.
    assert [len(r) for r in seen] == sorted(len(r) for r in seen)
    assert metrics["n_test"] == int((data["ts"] >= test_dates[0]).sum())
    assert len(metrics["fold_auc"]) == train.FOLDS


def test_walk_forward_refuses_too_little_history():
    data = train.build_dataset(_history(None, n=40))
    with pytest.raises(ValueError):
        train.walk_forward(data)


def test_walk_forward_refuses_a_fold_with_nothing_to_train_on():
    """A purge that empties fold 0 must fail with a clear message, not
    XGBoost's cryptic empty-DMatrix error."""
    data = train.build_dataset(_history(None, n=48))  # 10 labelled dates: all purged
    with pytest.raises(ValueError, match="no training rows"):
        train.walk_forward(data)


def test_no_history_at_all_is_a_clear_error():
    with pytest.raises(ValueError, match="no training history"):
        train.build_dataset({})


def test_noise_has_no_skill_and_a_planted_signal_does():
    """Sanity of the whole pipeline: pure noise scores ~0.5, a planted
    momentum regime is learned out of sample."""
    noise = train.walk_forward(train.build_dataset(_history(None)))
    signal = train.walk_forward(train.build_dataset(_history(trend_block=40)))
    assert abs(noise["auc"] - 0.5) < 0.05
    assert signal["auc"] > 0.65


def test_train_saves_a_loadable_version_with_its_metrics(tmp_path):
    returned = train.train(_history(trend_block=40), tmp_path, NOW)
    path = tmp_path / "20260922T120000Z"
    meta = json.loads((path / "meta.json").read_text())
    assert returned == meta
    assert meta["features"] == list(FEATURES)
    assert meta["symbols"] == ["AAA", "BBB", "CCC"]
    assert model.is_servable(meta["walk_forward"])
    loaded = model.load_latest(tmp_path)
    assert loaded.version == "20260922T120000Z"


def test_training_is_deterministic(tmp_path):
    history = _history(trend_block=40, n=300)
    version = train.train(history, tmp_path / "a", NOW)["version"]
    train.train(history, tmp_path / "b", NOW)
    first, second = (tmp_path / d / version / "model.json" for d in ("a", "b"))
    assert first.read_bytes() == second.read_bytes()


def test_fetch_history_routes_by_asset_class_and_drops_empty_symbols():
    client = MagicMock()
    client.get_daily_candles.side_effect = lambda s, *_: (
        candles_from_closes([1.0], symbol=s) if s == "AAPL" else []
    )
    client.get_crypto_daily_candles.side_effect = lambda s, *_: candles_from_closes(
        [1.0], symbol=s, asset_class="crypto"
    )
    with structlog.testing.capture_logs() as logs:
        history = train.fetch_history(client, ["AAPL", "EMPTY"], ["BTC-USD"], NOW)
    assert set(history) == {"AAPL", "BTC-USD"}
    dropped = [e for e in logs if e["event"] == "training_symbols_without_history"]
    assert [(e["log_level"], e["symbols"]) for e in dropped] == [("warning", ["EMPTY"])]
    assert client.get_crypto_daily_candles.call_args.args == ("BTC-USD", train.HISTORY_START, NOW)


def test_main_without_alpaca_keys_exits(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "")
    with pytest.raises(SystemExit):
        train.main([])
