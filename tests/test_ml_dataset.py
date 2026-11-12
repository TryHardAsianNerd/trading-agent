"""Directional-model dataset: feature values, no lookahead, labels."""

import math

import pytest
from tests.ml_fakes import candles_from_closes, random_walk

from trading_agent.ml.dataset import FEATURES, HORIZON_BARS, feature_frame, labelled_rows


def test_features_are_the_documented_scale_invariant_set():
    frame = feature_frame(candles_from_closes(random_walk(80, seed=1)))
    assert list(frame.columns) == [*FEATURES, "close"]
    # No volume feature: training and serving measure volume differently (D-024).
    # (vol_N is return volatility.)
    assert not {"volume", "obv", "vwap_20"} & set(frame.columns)


def test_known_values_on_the_last_bar():
    closes = [100.0 + i for i in range(40)]  # 100..139
    last = feature_frame(candles_from_closes(closes)).iloc[-1]
    assert last["ret_1"] == pytest.approx(math.log(139 / 138))
    assert last["ret_5"] == pytest.approx(math.log(139 / 134))
    assert last["ret_20"] == pytest.approx(math.log(139 / 119))
    assert last["dist_sma_10"] == pytest.approx(139 / 134.5 - 1)
    assert last["range_pct"] == pytest.approx((139 * 1.01 - 139 * 0.99) / 139, rel=1e-4)
    assert last["rsi_14"] == 1.0  # all gains, scaled to [0, 1]
    assert last["is_crypto"] == 0.0


def test_scale_invariance_prices_times_1000_give_identical_features():
    closes = random_walk(80, seed=2)
    base = feature_frame(candles_from_closes(closes))
    scaled = feature_frame(candles_from_closes([c * 1000 for c in closes]))
    for name in FEATURES:
        assert scaled[name].iloc[-1] == pytest.approx(base[name].iloc[-1], rel=1e-3, abs=1e-6)


def test_no_lookahead_future_bars_never_change_past_features():
    """The load-bearing property: a feature at bar t reads bars <= t only.
    Rewriting every bar after t must leave rows <= t bit-identical."""
    closes = random_walk(120, seed=3)
    cut = 80
    original = feature_frame(candles_from_closes(closes))
    tampered_closes = closes[:cut] + [c * 3 for c in closes[cut:]]
    tampered = feature_frame(candles_from_closes(tampered_closes))
    assert original.iloc[:cut].equals(tampered.iloc[:cut])
    assert not original.iloc[cut:].equals(tampered.iloc[cut:])


def test_lookbacks_fill_at_macd_34_bars():
    frame = feature_frame(candles_from_closes(random_walk(60, seed=4)))
    complete = frame[list(FEATURES)].notna().all(axis=1)
    assert not complete.iloc[:33].any()
    assert complete.iloc[33:].all()


def test_crypto_flag_and_empty_input():
    frame = feature_frame(candles_from_closes(random_walk(40, 5), asset_class="crypto"))
    assert (frame["is_crypto"] == 1.0).all()
    assert feature_frame([]).empty


def test_duplicate_timestamps_dedupe_last_wins():
    candles = candles_from_closes(random_walk(40, seed=6))
    dupe = candles[-1].model_copy(update={"close": candles[-1].close * 2})
    frame = feature_frame([*candles, dupe])
    assert len(frame) == 40
    assert frame["close"].iloc[-1] == float(dupe.close)


def test_zero_close_is_nan_never_inf():
    closes = random_walk(50, seed=7)
    closes[45] = 0.0
    frame = feature_frame(candles_from_closes(closes))
    values = frame[list(FEATURES)].to_numpy().ravel()
    assert not any(math.isinf(v) for v in values)


def test_labels_are_the_sign_of_the_forward_return():
    closes = random_walk(100, seed=8)
    rows = labelled_rows(candles_from_closes(closes), "AAA")
    frame = feature_frame(candles_from_closes(closes))
    for _, row in rows.iterrows():
        i = frame.index.get_loc(row["ts"])
        assert row["label"] == float(closes[i + HORIZON_BARS] > closes[i])
        assert row["label_end"] == frame.index[i + HORIZON_BARS]
    # The last HORIZON_BARS bars have no future bar: unlabelled, dropped.
    assert rows["ts"].max() == frame.index[-1 - HORIZON_BARS]
    assert rows["ts"].min() == frame.index[33]
    assert set(rows["symbol"]) == {"AAA"}
    assert rows[list(FEATURES)].notna().all().all()
