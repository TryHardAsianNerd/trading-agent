"""Directional-model artifacts: atomic versioned save, latest load, compatibility."""

import json

import pandas as pd
import pytest
import xgboost as xgb
from tests.ml_fakes import candles_from_closes, random_walk

from trading_agent.ml import model
from trading_agent.ml.dataset import FEATURES, HORIZON_BARS, labelled_rows


def _booster() -> xgb.Booster:
    rows = labelled_rows(candles_from_closes(random_walk(120, seed=1)), "AAA")
    matrix = xgb.DMatrix(rows[list(FEATURES)], label=rows["label"], feature_names=list(FEATURES))
    return xgb.train({"objective": "binary:logistic", "seed": 0}, matrix, num_boost_round=5)


def _meta(version: str, **overrides) -> dict:
    meta = {
        "version": version,
        "features": list(FEATURES),
        "horizon_bars": HORIZON_BARS,
        "walk_forward": {"auc": 0.55, "base_rate": 0.52, "fold_auc": [0.55] * 5},
    }
    return {**meta, **overrides}


def test_save_then_load_roundtrip_predicts_identically(tmp_path):
    booster = _booster()
    path = model.save(booster, _meta("20260901T000000Z"), tmp_path)
    assert {p.name for p in path.iterdir()} == {"model.json", "meta.json"}
    loaded = model.load_latest(tmp_path)
    assert loaded.version == "20260901T000000Z"
    assert loaded.oos_auc == 0.55
    rows = labelled_rows(candles_from_closes(random_walk(80, seed=2)), "AAA")
    expected = booster.predict(xgb.DMatrix(rows[list(FEATURES)], feature_names=list(FEATURES)))
    probs = loaded.prob_up(rows)
    assert probs == pytest.approx([float(p) for p in expected])
    # Plain floats: numpy.float64 breaks the checkpoint serializer.
    assert all(type(p) is float for p in probs)


def test_latest_version_wins_and_staging_dirs_are_ignored(tmp_path):
    booster = _booster()
    model.save(booster, _meta("20260901T000000Z"), tmp_path)
    model.save(booster, _meta("20260915T000000Z"), tmp_path)
    (tmp_path / ".tmp-abandoned").mkdir()  # a crashed training run's leftovers
    (tmp_path / "backup").mkdir()  # sorts after "2026..." but is not a version
    assert model.load_latest(tmp_path).version == "20260915T000000Z"


def test_no_model_yet_is_none(tmp_path):
    assert model.load_latest(tmp_path / "missing") is None
    assert model.load_latest(tmp_path) is None


def test_existing_version_is_never_overwritten(tmp_path):
    model.save(_booster(), _meta("20260901T000000Z"), tmp_path)
    with pytest.raises(FileExistsError):
        model.save(_booster(), _meta("20260901T000000Z"), tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["20260901T000000Z"]


@pytest.mark.parametrize(
    "override",
    [
        {"features": list(FEATURES)[:-1]},
        {"features": list(reversed(FEATURES))},
        {"horizon_bars": 1},
    ],
)
def test_model_from_another_feature_set_is_refused_not_substituted(tmp_path, override):
    """A newer incompatible model must not silently fall back to an older
    compatible one: the operator retrains, the cycle serves nothing meanwhile."""
    model.save(_booster(), _meta("20260901T000000Z"), tmp_path)
    model.save(_booster(), _meta("20260915T000000Z", **override), tmp_path)
    with pytest.raises(ValueError):
        model.load_latest(tmp_path)


def test_meta_is_written_as_json(tmp_path):
    path = model.save(_booster(), _meta("20260901T000000Z", extra=1), tmp_path)
    assert json.loads((path / "meta.json").read_text())["extra"] == 1


def test_prob_up_ignores_extra_columns_and_column_order(tmp_path):
    model.save(_booster(), _meta("20260901T000000Z"), tmp_path)
    loaded = model.load_latest(tmp_path)
    rows = labelled_rows(candles_from_closes(random_walk(80, seed=3)), "AAA")
    shuffled = rows[["symbol", *reversed(FEATURES)]]
    assert loaded.prob_up(shuffled) == loaded.prob_up(pd.DataFrame(rows[list(FEATURES)]))


@pytest.mark.parametrize(
    ("pooled", "folds", "servable"),
    [
        (0.53, [0.53, 0.50, 0.56, 0.56, 0.50], True),  # median 0.53
        (0.53, [0.49, 0.50, 0.58, 0.51, 0.50], False),  # one lucky fold carries the pool
        (0.52, [0.60] * 5, False),  # the floor itself is not skill
        (0.60, [0.521] * 5, True),
    ],
)
def test_serving_needs_pooled_and_median_fold_auc_above_the_floor(pooled, folds, servable):
    assert model.is_servable({"auc": pooled, "fold_auc": folds}) is servable
