"""ML predictor node: serves the newest model, degrades to empty, never crashes."""

import contextlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import structlog
from tests.ml_fakes import START, candles_from_closes, random_walk

from trading_agent.graph.deps import CycleDeps
from trading_agent.graph.nodes import ml_predictor
from trading_agent.ml import model, train

N_BARS = 120
# The cycle runs on the day after the last stored bar... plus a still-forming bar for today.
TODAY = START + timedelta(days=N_BARS)


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    history = {
        sym: candles_from_closes(random_walk(700, seed, 40), symbol=sym)
        for seed, sym in enumerate(["AAA", "BBB", "CCC"])
    }
    train.train(history, root, datetime(2026, 9, 1, tzinfo=UTC))
    return root


class _Repo:
    """Stands in for OhlcvRepository: serves canned candles per symbol."""

    candles: dict = {}  # noqa: RUF012 - set per test

    def __init__(self, session) -> None:
        pass

    def recent_daily_candles(self, symbol, limit=60):
        return _Repo.candles.get(symbol, [])[-limit:]


def _deps(model_dir) -> CycleDeps:
    return CycleDeps(
        settings=None,
        session_factory=contextlib.nullcontext,
        brokers=[],
        kill_switch=None,
        tracker=None,
        analyst=None,
        coingecko=None,
        model_dir=model_dir,
    )


def _state(*symbols: str) -> dict:
    return {
        "cycle_id": "c1",
        "cycle_started_at": TODAY + timedelta(hours=15),
        "market_data": {"prices": {s: Decimal("1") for s in symbols}},
    }


@pytest.fixture
def repo(monkeypatch):
    closes = random_walk(N_BARS + 1, seed=9)
    _Repo.candles = {
        # N_BARS complete bars plus today's still-forming bar, which must be ignored.
        "AAA": candles_from_closes(closes),
        "BTC-USD": candles_from_closes(closes, symbol="BTC-USD", asset_class="crypto"),
        # 40 complete bars: every feature is filled, but the indicators are not warmed up.
        "NEW": candles_from_closes(closes[:40], symbol="NEW", start=TODAY - timedelta(days=40)),
        # Plenty of history, but the newest bar is 5 days old.
        "OLD": candles_from_closes(closes, symbol="OLD", start=START - timedelta(days=5)),
    }
    monkeypatch.setattr(ml_predictor, "OhlcvRepository", _Repo)
    return _Repo


def test_predictions_for_every_symbol_with_enough_complete_history(trained_dir, repo):
    with structlog.testing.capture_logs() as logs:
        out = ml_predictor.predict(_state("AAA", "BTC-USD", "NEW", "OLD"), deps=_deps(trained_dir))
    preds = out["predictions"]
    skipped = {e["symbol"]: e["reason"] for e in logs if e["event"] == "ml_prediction_skipped"}
    assert skipped == {"NEW": "history_too_short", "OLD": "stale_bars"}
    loaded = model.load_latest(trained_dir)
    assert preds["model"] == {
        "version": loaded.version,
        "horizon_bars": 5,
        "oos_auc": loaded.oos_auc,
        "base_rate": loaded.meta["walk_forward"]["base_rate"],
    }
    assert set(preds["by_symbol"]) == {"AAA", "BTC-USD"}
    last_complete = (TODAY - timedelta(days=1)).date().isoformat()
    for p in preds["by_symbol"].values():
        assert 0.0 <= p["prob_up"] <= 1.0
        assert type(p["prob_up"]) is float  # checkpoint serializer refuses numpy floats
        assert p["as_of"] == last_complete  # today's forming bar is excluded


def test_todays_forming_bar_does_not_change_the_prediction(trained_dir, repo):
    before = ml_predictor.predict(_state("AAA"), deps=_deps(trained_dir))
    forming = repo.candles["AAA"][-1]
    repo.candles["AAA"][-1] = forming.model_copy(update={"close": forming.close * 5})
    after = ml_predictor.predict(_state("AAA"), deps=_deps(trained_dir))
    assert before == after


def test_no_model_yet_emits_empty_predictions(tmp_path, repo):
    with structlog.testing.capture_logs() as logs:
        out = ml_predictor.predict(_state("AAA"), deps=_deps(tmp_path))
    assert out == {"predictions": {}}
    assert [e["log_level"] for e in logs if e["event"] == "ml_model_missing"] == ["warning"]


def test_model_below_the_skill_floor_is_not_served(tmp_path, repo, trained_dir):
    loaded = model.load_latest(trained_dir)
    weak_metrics = {**loaded.meta["walk_forward"], "fold_auc": [0.50] * 5}
    weak = {**loaded.meta, "version": "20260902T000000Z", "walk_forward": weak_metrics}
    model.save(loaded.booster, weak, tmp_path)
    with structlog.testing.capture_logs() as logs:
        out = ml_predictor.predict(_state("AAA"), deps=_deps(tmp_path))
    assert out == {"predictions": {}}
    assert any(e["event"] == "ml_model_below_skill_floor" for e in logs)


def test_incompatible_model_degrades_to_empty(tmp_path, repo, trained_dir):
    loaded = model.load_latest(trained_dir)
    model.save(loaded.booster, {**loaded.meta, "horizon_bars": 1}, tmp_path)
    with structlog.testing.capture_logs() as logs:
        out = ml_predictor.predict(_state("AAA"), deps=_deps(tmp_path))
    assert out == {"predictions": {}}
    assert [e["log_level"] for e in logs if e["event"] == "ml_model_unloadable"] == ["warning"]


def test_database_failure_is_contained_and_pages(trained_dir, monkeypatch):
    class _Broken:
        def __init__(self, session):
            raise ConnectionError("db down")

    monkeypatch.setattr(ml_predictor, "OhlcvRepository", _Broken)
    with structlog.testing.capture_logs() as logs:
        out = ml_predictor.predict(_state("AAA"), deps=_deps(trained_dir))
    assert out == {"predictions": {}}
    assert [e["log_level"] for e in logs if e["event"] == "ml_predictions_unavailable"] == ["error"]


def test_nothing_priced_serves_an_empty_symbol_map(trained_dir, repo):
    out = ml_predictor.predict(_state(), deps=_deps(trained_dir))
    assert out["predictions"]["by_symbol"] == {}


@pytest.mark.parametrize(
    ("asset_class", "age_days", "served"),
    [
        ("crypto", 1, True),
        ("crypto", 2, False),
        ("stock", 4, True),  # Friday's bar on a Tuesday after a Monday holiday
        ("stock", 5, False),
    ],
)
def test_bar_age_limit_per_asset_class(trained_dir, repo, asset_class, age_days, served):
    closes = random_walk(N_BARS, seed=9)
    start = TODAY - timedelta(days=N_BARS - 1 + age_days)
    repo.candles["X"] = candles_from_closes(
        closes, symbol="X", asset_class=asset_class, start=start
    )
    out = ml_predictor.predict(_state("X"), deps=_deps(trained_dir))
    assert ("X" in out["predictions"]["by_symbol"]) is served


def test_one_symbols_bad_candles_do_not_cost_the_others(trained_dir, repo, monkeypatch):
    real = ml_predictor.feature_frame

    def flaky(candles):
        if candles[0].symbol == "BTC-USD":
            raise ValueError("malformed")
        return real(candles)

    monkeypatch.setattr(ml_predictor, "feature_frame", flaky)
    with structlog.testing.capture_logs() as logs:
        out = ml_predictor.predict(_state("AAA", "BTC-USD"), deps=_deps(trained_dir))
    assert set(out["predictions"]["by_symbol"]) == {"AAA"}
    assert not [e for e in logs if e["log_level"] == "error"]


def test_predictions_survive_the_checkpoint_serializer(trained_dir, repo):
    """Predictions live in checkpointed state: numpy/pandas types there would
    fail the cycle at the next checkpoint write."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    preds = ml_predictor.predict(_state("AAA", "BTC-USD"), deps=_deps(trained_dir))
    serde = JsonPlusSerializer()
    assert serde.loads_typed(serde.dumps_typed(preds)) == preds


@pytest.mark.parametrize(("trained_days_ago", "warned"), [(8, False), (9, True)])
def test_an_old_model_is_still_served_but_warns(
    tmp_path, repo, trained_dir, trained_days_ago, warned
):
    """The weekly retrain stopping is invisible otherwise: the model keeps
    serving, but every cycle says how old it is."""
    loaded = model.load_latest(trained_dir)
    started = TODAY + timedelta(hours=15)
    trained_at = (started - timedelta(days=trained_days_ago)).isoformat()
    model.save(loaded.booster, {**loaded.meta, "trained_at": trained_at}, tmp_path)
    with structlog.testing.capture_logs() as logs:
        out = ml_predictor.predict(_state("AAA"), deps=_deps(tmp_path))
    assert "AAA" in out["predictions"]["by_symbol"]
    stale = [e for e in logs if e["event"] == "ml_model_stale"]
    assert bool(stale) is warned
    assert all(e["log_level"] == "warning" for e in stale)
