"""Backtest: the simulated day through the live gate, metrics, and the report."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from tests.ml_fakes import candles_from_closes, random_walk

from trading_agent.backtest import engine
from trading_agent.backtest import study as study_mod
from trading_agent.backtest.engine import (
    ClosedTrade,
    Params,
    SimRedis,
    StopRule,
    backtest_limits,
    simulate,
)
from trading_agent.backtest.metrics import buy_and_hold, exposure_matched, summarize
from trading_agent.backtest.run import format_table, run
from trading_agent.data.models import Candle

START = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday
WARMUP = 30  # bars before the first score: the gate needs ATR-14 and 20 liquidity bars
# The engine tests pin 2x ATR, a 3% trail and a 5% cap rather than the live limits;
# that keeps every expected price hand-checkable.
LIMITS = replace(
    backtest_limits(),
    atr_multiplier=Decimal("2.0"),
    trailing_stop_pct=Decimal("0.03"),
    max_position_pct=Decimal("0.05"),
)


def _bars(
    rows: list[tuple[float, float, float, float]], symbol: str = "AAA", asset_class="crypto"
) -> list[Candle]:
    """Daily candles from START, one per OHLC row, with volume that clears the liquidity filter."""
    return [
        Candle(
            symbol=symbol,
            asset_class=asset_class,
            ts=START + timedelta(days=i),
            open=Decimal(str(o)),
            high=Decimal(str(h)),
            low=Decimal(str(lo)),
            close=Decimal(str(c)),
            volume=Decimal(10_000_000),
            source="itest",
        )
        for i, (o, h, lo, c) in enumerate(rows)
    ]


def _flat(n: int, price: float = 100.0) -> list[tuple[float, float, float, float]]:
    """`n` quiet OHLC days around `price` (ATR about 2)."""
    return [(price, price + 1, price - 1, price)] * n


def _scores(symbol: str, day_scores: dict[int, float]) -> pd.DataFrame:
    """Scores shaped like run()'s (base_rate 0.5), keyed by day offset from START."""
    return pd.DataFrame(
        {
            "ts": [pd.Timestamp(START + timedelta(days=d)) for d in day_scores],
            "symbol": symbol,
            "score": list(day_scores.values()),
            "base_rate": 0.5,
        }
    )


def _p(edge: float, slippage: Decimal, crypto_fee: Decimal = Decimal(0)) -> Params:
    """Params with no crypto fee unless asked, so prices in tests stay round."""
    return Params(edge, slippage, crypto_fee=crypto_fee)


def _neutral(days: range) -> dict[int, float]:
    """A base-rate score of 0.5 (no signal) for each day offset."""
    return dict.fromkeys(days, 0.5)


def test_a_buy_signal_fills_next_open_and_a_stop_closes_it():
    """Signal at day-30's close -> fill at day-31's open (+slippage); the stop
    is hit on day 33's low -> exit at the stop (-slippage)."""
    rows = [
        *_flat(WARMUP + 1),
        (101, 102, 99, 100),
        (100, 101, 99, 100),
        (100, 100, 80, 85),
        *_flat(5, 85),
    ]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 9)), WARMUP: 0.9})
    result = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal("0.01")), LIMITS)

    (trade,) = result.trades
    assert trade.entry_day == (START + timedelta(days=WARMUP + 1)).date()
    assert trade.entry_price == Decimal("101") * Decimal("1.01")
    assert trade.reason == "stop"
    assert trade.exit_day == (START + timedelta(days=WARMUP + 3)).date()
    # ATR stop 100 - 2 x ATR(2) = 96, but the 3% trail from the 102.01 entry (98.95)
    # is higher, and the stop only ever takes the higher of the two.
    assert trade.exit_price == Decimal("102.01") * Decimal("0.97") * Decimal("0.99")
    # Sizing came from the live gate: at most 5% of equity at the signal price.
    assert trade.quantity * Decimal(100) <= Decimal(100_000) * Decimal("0.05")


def test_crypto_pays_the_fee_on_both_sides_and_stocks_do_not():
    rows = [*_flat(WARMUP + 2), (103, 104, 102, 103), *_flat(3, 103)]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 5)), WARMUP: 0.9, WARMUP + 2: 0.1})
    fee = Decimal("0.006")
    crypto = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0), fee), LIMITS)
    (trade,) = crypto.trades
    assert trade.entry_price == Decimal(100) * (1 + fee)
    assert trade.exit_price == Decimal(103) * (1 - fee)
    stock = simulate(
        {"AAA": _bars(rows, asset_class="stock")}, scores, _p(0.1, Decimal(0), fee), LIMITS
    )
    assert stock.trades[0].entry_price == Decimal(100)


def test_exposure_is_the_invested_share_of_equity():
    rows = _flat(WARMUP + 4)
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 4)), WARMUP: 0.9})
    result = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0)), LIMITS)
    held = result.exposure[pd.Timestamp((START + timedelta(days=WARMUP + 1)).date())]
    assert 0 < held <= 0.05 + 1e-9  # the gate's 5% cap
    assert result.exposure.iloc[0] == 0


def test_a_gap_through_the_stop_fills_at_the_open():
    rows = [*_flat(WARMUP + 2), (70, 72, 60, 65), *_flat(3, 65)]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 6)), WARMUP: 0.9})
    result = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0)), LIMITS)
    assert result.trades[0].exit_price == Decimal(70)  # worse than the 96 stop


def test_the_trail_ratchets_from_earlier_highs_but_not_the_same_day():
    """After a run to 120, the 3% trail (116.4) outranks the ATR stop."""
    rise = [(100 + 5 * i, 101 + 5 * i, 99 + 5 * i, 100 + 5 * i) for i in range(1, 5)]
    rows = _flat(WARMUP + 1) + rise + [(119, 119, 116, 117)] + _flat(3, 117)
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 9)), WARMUP: 0.9})
    result = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0)), LIMITS)
    trade = result.trades[0]
    assert trade.reason == "stop"
    assert trade.exit_price == Decimal("121") * Decimal("0.97")  # high 121 on the last rise day


def test_a_days_own_high_never_raises_that_days_stop():
    """Day 36 prints a new high of 130 and a low of 120: with the trail
    ratcheted from its own high (126.1) that would stop out; measured from the
    previous high (121 -> 117.37) it must not."""
    rise = [(100 + 5 * i, 101 + 5 * i, 99 + 5 * i, 100 + 5 * i) for i in range(1, 5)]
    rows = [*_flat(WARMUP + 1), *rise, (121, 130, 120, 125), *_flat(3, 128)]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 9)), WARMUP: 0.9})
    result = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0)), LIMITS)
    assert result.trades == []  # still held; later lows (127) sit above 130 x 0.97 = 126.1


def test_a_sell_signal_closes_at_the_next_open():
    rows = _flat(WARMUP + 6)
    rows[WARMUP + 3] = (103, 104, 102, 103)
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 6)), WARMUP: 0.9, WARMUP + 2: 0.1})
    result = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0)), LIMITS)
    (trade,) = result.trades
    assert (trade.reason, trade.exit_price) == ("signal", Decimal(103))


def test_a_stock_order_waits_over_the_weekend():
    """Friday's signal fills at Monday's open: no bar, no fill."""
    stock = [c for c in _bars(_flat(WARMUP + 20), "SSS", "stock") if c.ts.weekday() < 5]
    friday = next(c.ts for c in stock[WARMUP:] if c.ts.weekday() == 4)
    monday = friday + timedelta(days=3)
    stock = [
        c.model_copy(update={"high": Decimal(111), "close": Decimal(110)}) if c.ts == monday else c
        for c in stock
    ]
    offset = (friday - START).days
    scores = _scores("SSS", {**_neutral(range(offset, offset + 5)), offset: 0.9})
    crypto = _bars(_flat(WARMUP + 20), "CCC")  # keeps the calendar ticking over the weekend
    result = simulate({"SSS": stock, "CCC": crypto}, scores, _p(0.1, Decimal(0)), LIMITS)
    for weekend_day in (friday + timedelta(days=1), friday + timedelta(days=2)):
        assert result.equity[pd.Timestamp(weekend_day.date())] == 100_000  # nothing held yet
    assert result.equity[pd.Timestamp(monday.date())] > 100_000  # bought at 100, closed 110


def test_drawdown_kill_switch_halts_every_entry(monkeypatch):
    """Once equity is 15% off its peak the live gate arms the kill switch and
    no entry is ever approved for the rest of the run."""
    monkeypatch.setattr(engine, "_portfolio", _crashed_portfolio(engine._portfolio))
    rows = _flat(WARMUP + 10)
    scores = _scores("AAA", dict.fromkeys(range(WARMUP, WARMUP + 10), 0.9))
    result = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0)), LIMITS)
    assert result.halted_on == (START + timedelta(days=WARMUP)).date()
    assert result.trades == []
    assert result.rejections["max_drawdown_exceeded"] >= 1


def _crashed_portfolio(real):
    """Wrap engine._portfolio so every book reads 20% below its peak."""

    def wrapped(*args):
        """Return the real state with a peak 25% above equity."""
        state = real(*args)
        return state.model_copy(update={"peak_equity": state.equity * Decimal("1.25")})

    return wrapped


def test_the_gate_window_never_includes_a_future_bar():
    candles = _bars(_flat(100))
    days = [c.ts.date() for c in candles]
    day = days[70]
    window = engine._window(candles, days, day)
    assert window[-1].ts.date() == day
    assert len(window) == engine.GATE_CANDLES


def test_sim_redis_expires_on_the_simulated_clock():
    redis = SimRedis()
    redis.now = datetime(2024, 1, 1, tzinfo=UTC)
    redis.set("cool", "x", ex=3600)
    redis.set("kill", "y")
    redis.now += timedelta(minutes=59)
    assert redis.get("cool") == "x"
    redis.now += timedelta(minutes=1)
    assert redis.get("cool") is None
    assert redis.get("kill") == "y"
    redis.delete("kill")
    assert redis.get("kill") is None


def test_backtest_limits_only_relax_the_crypto_liquidity_floor():
    live, bt = engine.DEFAULT_LIMITS, backtest_limits()
    assert bt.min_avg_dollar_volume_crypto == 0
    assert replace(bt, min_avg_dollar_volume_crypto=live.min_avg_dollar_volume_crypto) == live


def _trade(pnl: int) -> ClosedTrade:
    """A one-unit round trip with the given P&L (negative is a loss)."""
    return ClosedTrade(
        "A",
        date(2024, 1, 1),
        date(2024, 1, 2),
        Decimal(1),
        Decimal(100),
        Decimal(100 + pnl),
        "stop",
    )


def test_summary_metrics_known_values():
    days = pd.date_range("2024-01-01", periods=366, freq="D")
    equity = pd.Series([100.0] * 100 + [80.0] * 100 + [121.0] * 166, index=days)
    m = summarize(equity, 100.0, [_trade(30), _trade(-10), _trade(-20)])
    assert m["total_return"] == pytest.approx(0.21)
    assert m["cagr"] == pytest.approx(0.21, abs=1e-3)  # exactly one year
    assert m["max_drawdown"] == pytest.approx(0.2)
    assert m["trades"] == 3
    assert m["win_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["profit_factor"] == pytest.approx(1.0)
    assert summarize(equity, 100.0, [_trade(5)])["profit_factor"] is None
    assert "trades" not in summarize(equity, 100.0)


def test_returns_count_from_the_starting_cash_not_the_first_close():
    """A benchmark's first mark is already after its entry costs and day-one
    move: measuring from it would hide both."""
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    m = summarize(pd.Series([99.0, 99.0, 110.0], index=days), 100.0)
    assert m["total_return"] == pytest.approx(0.10)


def test_buy_and_hold_splits_equally_and_pays_slippage_once():
    a = _bars([(100, 100, 100, 100), (100, 100, 100, 200)], "A")
    b = _bars([(50, 50, 50, 50), (50, 50, 50, 50)], "B")
    end = (START + timedelta(days=1)).date()
    curve = buy_and_hold(
        {"A": a, "B": b}, START.date(), end, Decimal("0.01"), Decimal(1000), crypto_fee=Decimal(0)
    )
    assert curve.iloc[0] == pytest.approx(1000 / 1.01)
    assert curve.iloc[-1] == pytest.approx((1000 + 500) / 1.01)
    with_fee = buy_and_hold(
        {"A": a, "B": b}, START.date(), end, Decimal(0), Decimal(1000), crypto_fee=Decimal("0.006")
    )
    assert with_fee.iloc[0] == pytest.approx(1000 / 1.006)


def test_a_symbol_is_cash_until_its_first_bar_never_a_future_price():
    """A Saturday start must not value Monday-listed stock at Monday's close."""
    stock = [c.model_copy(update={"asset_class": "stock"}) for c in _bars(_flat(5), "S")][2:]
    stock[0] = stock[0].model_copy(update={"close": Decimal(200)})
    curve = buy_and_hold(
        {"S": stock}, START.date(), (START + timedelta(days=4)).date(), Decimal(0), Decimal(1000)
    )
    assert list(curve.iloc[:2]) == [1000, 1000]
    assert curve.iloc[2] == 2000


def test_exposure_matched_benchmark():
    curve = pd.Series([100.0, 200.0])
    assert list(exposure_matched(curve, 100.0, 0.25)) == [100.0, 125.0]


def test_run_reports_every_config_and_the_benchmark():
    history = {
        sym: candles_from_closes(random_walk(500, seed, 40), symbol=sym, asset_class="crypto")
        for seed, sym in enumerate(["AAA", "BBB"])
    }
    report = run(history, [0.0, 0.05], [Decimal("0.001"), Decimal("0.005")])
    assert len(report["runs"]) == 4 and len(report["benchmark"]) == 2
    assert len(report["fold_base_rates"]) == 5
    assert all(0 < r < 1 for r in report["fold_base_rates"])
    assert report["window"]["start"] > "2022-01-03"  # second half of the timeline only
    for r in report["runs"]:
        assert {"total_return", "sharpe", "max_drawdown", "trades", "mean_exposure"} <= set(
            r["metrics"]
        )
        assert "total_return" in r["matched_benchmark"]
    table = format_table(report)
    assert "edge 0.05 slip 0.005" in table and "buy&hold" in table


# --- stop rules and the study (D-031) ---------------------------------------


def test_stop_rule_trail_levels():
    high, atr = Decimal(100), Decimal(2)
    assert StopRule(trail_pct=Decimal("0.03")).trail(high, atr) == Decimal("97.00")
    assert StopRule(trail_atr=Decimal(3)).trail(high, atr) == Decimal(94)
    assert StopRule().trail(high, atr) is None


def test_atr_stop_only_rides_through_a_dip_the_3pct_trail_would_sell():
    """Entry at 100 with ATR 2: the 2x ATR stop is 96, the live 3% trail from
    the 101 high is 97.97. A dip to 97.5 sells under the trail, not without it."""
    rows = [*_flat(WARMUP + 2), (100, 101, 97.5, 100), *_flat(3, 100)]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 6)), WARMUP: 0.9})
    live = simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0)), LIMITS)
    wide = Params(0.1, Decimal(0), crypto_fee=Decimal(0), stops=StopRule())
    atr_only = simulate({"AAA": _bars(rows)}, scores, wide, LIMITS)
    assert [t.reason for t in live.trades] == ["stop"]
    assert atr_only.trades == []


def test_a_close_triggered_trail_ignores_a_wick_and_exits_at_the_next_open():
    rows = [
        *_flat(WARMUP + 2),
        (100, 100, 96.5, 99.5),
        (99, 99, 96.5, 96.8),
        (96, 97, 95, 96),
        *_flat(2, 96),
    ]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 7)), WARMUP: 0.9})
    rule = StopRule(trail_pct=Decimal("0.03"), trail_on_close=True)  # 101 high -> 97.97
    params = Params(0.1, Decimal(0), crypto_fee=Decimal(0), stops=rule)
    (trade,) = simulate({"AAA": _bars(rows)}, scores, params, LIMITS).trades
    # The wick to 96.5 closed at 99.5: held. The next day closed at 96.8, under the
    # trail: out at the following open (96) - a close is not tradable once observed.
    assert trade.exit_day == (START + timedelta(days=WARMUP + 4)).date()
    assert trade.exit_price == Decimal(96)


def test_the_atr_stop_stays_intraday_under_a_close_trail():
    """A crash through the ATR stop exits the same day at the stop, even when
    the trail waits for the close."""
    rows = [*_flat(WARMUP + 2), (100, 100, 90, 99), *_flat(3, 99)]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 6)), WARMUP: 0.9})
    rule = StopRule(trail_pct=Decimal("0.03"), trail_on_close=True)
    params = Params(0.1, Decimal(0), crypto_fee=Decimal(0), stops=rule)
    (trade,) = simulate({"AAA": _bars(rows)}, scores, params, LIMITS).trades
    assert trade.exit_day == (START + timedelta(days=WARMUP + 2)).date()
    assert trade.exit_price == Decimal(96)  # 100 - 2 x ATR 2


def test_an_atr_trail_uses_the_entry_atr_the_gate_stopped_on():
    """3 x ATR(2) trail below a 121 high = 115: recovering the entry ATR from
    the gate's stop must reproduce the gate's own ATR."""
    rise = [(100 + 5 * i, 101 + 5 * i, 99 + 5 * i, 100 + 5 * i) for i in range(1, 5)]
    rows = [*_flat(WARMUP + 1), *rise, (118, 118, 113, 115), *_flat(3, 115)]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 9)), WARMUP: 0.9})
    params = Params(0.1, Decimal(0), crypto_fee=Decimal(0), stops=StopRule(trail_atr=Decimal(3)))
    (trade,) = simulate({"AAA": _bars(rows)}, scores, params, LIMITS).trades
    assert trade.exit_price == Decimal(121) - 3 * Decimal(2)  # high 121, entry ATR 2


def test_no_rule_means_the_limits_own_trail():
    """Params.stops=None follows limits.trailing_stop_pct: a limits change must
    never be silently ignored by the backtest."""
    rows = [*_flat(WARMUP + 2), (100, 101, 97.5, 100), *_flat(3, 100)]
    scores = _scores("AAA", {**_neutral(range(WARMUP, WARMUP + 6)), WARMUP: 0.9})
    no_trail = replace(LIMITS, trailing_stop_pct=None)
    assert simulate({"AAA": _bars(rows)}, scores, _p(0.1, Decimal(0)), no_trail).trades == []


def test_trend_scores_read_no_future_bar():
    closes = random_walk(200, seed=5)
    history = {"AAA": candles_from_closes(closes)}
    days = pd.Series(
        pd.date_range(START.date() + timedelta(days=60), periods=100, freq="D", tz=UTC)
    )
    base = study_mod.trend_scores(history, days)
    tampered = {"AAA": candles_from_closes(closes[:120] + [c * 5 for c in closes[120:]])}
    changed = study_mod.trend_scores(tampered, days)
    cut = pd.Timestamp(candles_from_closes(closes)[119].ts)
    before = base[base["ts"] <= cut].reset_index(drop=True)
    assert before.equals(changed[changed["ts"] <= cut].reset_index(drop=True))
    assert set(base["score"]) <= {0.0, 1.0}


def test_study_ranks_on_tune_and_reports_validate(monkeypatch):
    monkeypatch.setattr(study_mod, "ATR_MULTIPLES", (Decimal("2.0"),))
    monkeypatch.setattr(study_mod, "LOSS_STREAKS", (5,))
    monkeypatch.setattr(study_mod, "ATR_MULTIPLES", (Decimal("2.0"), Decimal("3.0")))
    monkeypatch.setattr(
        study_mod,
        "STOP_RULES",
        {
            study_mod.PRE_D032: StopRule(trail_pct=Decimal("0.03")),
            "ATR stop only": StopRule(),
        },
    )
    history = {
        sym: candles_from_closes(random_walk(500, seed, 40), symbol=sym, asset_class="crypto")
        for seed, sym in enumerate(["AAA", "BBB"])
    }
    report = study_mod.study(history)
    assert len(report["runs"]) == 3 * 2 * 2 * 2  # signals x ATR x caps x stops
    sharpes = [r["tune"]["sharpe"] for r in report["runs"]]
    assert sharpes == sorted(sharpes, reverse=True)
    assert report["tune"][1] < report["validate"][0]
    assert all({"bh_at_exposure", "mean_exposure"} <= set(r["validate"]) for r in report["runs"])
    table = study_mod.format_table(report)
    assert "<- pre-D-032 limits" in table and "<- D-032 limits" in table
    by_rule = {
        (r["signal"], r["atr_multiple"], r["position_cap"], r["stops"]): r["validate"]
        for r in report["runs"]
    }
    # The 3% trail row must really trail: it cannot equal the ATR-only row.
    assert (
        by_rule[("always", "2.0", "0.05", study_mod.PRE_D032)]
        != by_rule[("always", "2.0", "0.05", "ATR stop only")]
    )


def test_the_tune_half_never_sees_a_validate_bar(monkeypatch):
    seen = []
    real = study_mod.simulate

    def spy(history, scores, params, limits):
        """Record each half's last bar and first score, then simulate."""
        seen.append((max(c.ts for cs in history.values() for c in cs), scores["ts"].min()))
        return real(history, scores, params, limits)

    monkeypatch.setattr(study_mod, "simulate", spy)
    monkeypatch.setattr(study_mod, "ATR_MULTIPLES", (Decimal("2.0"),))
    monkeypatch.setattr(study_mod, "LOSS_STREAKS", (5,))
    monkeypatch.setattr(study_mod, "POSITION_CAPS", (Decimal("0.05"),))
    monkeypatch.setattr(study_mod, "STOP_RULES", {study_mod.PRE_D032: StopRule()})
    history = {
        sym: candles_from_closes(random_walk(500, seed, 40), symbol=sym, asset_class="crypto")
        for seed, sym in enumerate(["AAA", "BBB"])
    }
    report = study_mod.study(history)
    tune_end = pd.Timestamp(report["tune"][1], tz=UTC)
    tune_runs, validate_runs = seen[0::2], seen[1::2]
    assert all(last_bar <= tune_end for last_bar, _ in tune_runs)
    assert all(first_score > tune_end for _, first_score in validate_runs)


def test_the_live_trend_label_matches_the_backtest_signal_every_day():
    """D-038: the feature engine's trend_up, fed each day's complete bars, must
    equal the study's trend score for that day - the live rule IS the tested one."""
    from trading_agent.features.engine import trend_up

    candles = candles_from_closes(random_walk(160, seed=11, trend_block=20))
    days = pd.Series([pd.Timestamp(c.ts) for c in candles[49:]])
    scores = study_mod.trend_scores({"AAA": candles}, days).set_index("ts")["score"]
    live = {pd.Timestamp(c.ts): trend_up(candles[: i + 1]) for i, c in enumerate(candles)}
    assert len(scores) == len(days)
    assert {0.0, 1.0} <= set(scores)  # both labels exercised
    assert all(live[ts] == score for ts, score in scores.items())
