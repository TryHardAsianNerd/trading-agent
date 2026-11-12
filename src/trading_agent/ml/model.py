"""Directional-model versions: atomic save, latest-version load, inference.

Layout: MODEL_DIR/<UTC training timestamp>/{model.json, meta.json}. A version is
written to a temp directory and renamed into place, so the cycle never loads a half
written model. model.json is XGBoost's JSON format, never pickle (which runs code on
load). A model whose features or horizon differ from this code's must be retrained.
"""

import json
import os
import re
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb

from trading_agent.ml.dataset import FEATURES, HORIZON_BARS

# Relative to the working directory (the repo root for `python -m ...`); gitignored.
MODEL_DIR = Path("models") / "directional"
# Walk-forward AUC a model must exceed before the cycle serves its predictions.
MIN_OOS_AUC = 0.52
# Only directories named like a version count: a stray "backup/" must not become "latest".
_VERSION = re.compile(r"^\d{8}T\d{6}Z$")


def is_servable(walk_forward: dict) -> bool:
    """Whether both the pooled and the median fold AUC exceed MIN_OOS_AUC.

    Pooled AUC alone can credit calibration shifts between folds as skill, and one
    lucky fold can carry the pool.
    """
    median_fold = statistics.median(walk_forward["fold_auc"])
    return walk_forward["auc"] > MIN_OOS_AUC and median_fold > MIN_OOS_AUC


@dataclass(frozen=True)
class DirectionalModel:
    """A loaded model version: the booster and its meta (features, horizon, metrics)."""

    booster: xgb.Booster
    meta: dict

    @property
    def version(self) -> str:
        """The version id, the UTC training timestamp ("20260923T030539Z")."""
        return self.meta["version"]

    @property
    def oos_auc(self) -> float:
        """Pooled walk-forward out-of-sample AUC recorded at training (0.5 = no skill)."""
        return float(self.meta["walk_forward"]["auc"])

    def prob_up(self, rows: pd.DataFrame) -> list[float]:
        """Probability that the close rises over the horizon, for each row of features."""
        matrix = xgb.DMatrix(rows[list(FEATURES)], feature_names=list(FEATURES))
        return [float(p) for p in self.booster.predict(matrix)]


def save(booster: xgb.Booster, meta: dict, root: Path = MODEL_DIR) -> Path:
    """Write a model version atomically and return its directory; FileExistsError if it exists."""
    root.mkdir(parents=True, exist_ok=True)
    final = root / meta["version"]
    if final.exists():
        raise FileExistsError(f"model version {final.name} already exists")
    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=root))
    booster.save_model(staging / "model.json")
    (staging / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.replace(staging, final)
    return final


def load_latest(root: Path = MODEL_DIR) -> DirectionalModel | None:
    """Load the newest model version, or None when none exists.

    Raises ValueError when its features or horizon differ; an older one is never used instead.
    """
    if not root.is_dir():
        return None
    versions = sorted(d for d in root.iterdir() if d.is_dir() and _VERSION.match(d.name))
    if not versions:
        return None
    latest = versions[-1]
    meta = json.loads((latest / "meta.json").read_text())
    if meta.get("features") != list(FEATURES) or meta.get("horizon_bars") != HORIZON_BARS:
        raise ValueError(f"model {latest.name} predates the current feature set")
    booster = xgb.Booster()
    booster.load_model(latest / "model.json")
    return DirectionalModel(booster=booster, meta=meta)
