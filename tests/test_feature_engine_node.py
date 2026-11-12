"""Feature engine node: the trend label reads complete bars only (D-038)."""

import contextlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from trading_agent.graph.deps import CycleDeps
from trading_agent.graph.nodes import feature_engine

NOW = datetime(2026, 9, 23, 15, tzinfo=UTC)


def test_trend_up_reads_only_bars_before_the_cycle_day(monkeypatch):
    """Today's still-forming bar must never reach the trend label (D-038):
    it would flip the label intraday, which the backtest never does."""
    today = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    bars = [SimpleNamespace(ts=today - timedelta(days=d)) for d in (2, 1, 0)]
    seen = []

    class _Bars:
        def recent_daily_candles(self, symbol, limit):
            """The three bars above."""
            return bars

    def fake_trend(complete):
        seen.append([b.ts for b in complete])
        return 1.0

    monkeypatch.setattr(feature_engine, "snapshot", lambda candles: {"close": 1.0})
    monkeypatch.setattr(feature_engine, "OhlcvRepository", lambda s: _Bars())
    monkeypatch.setattr(feature_engine, "_add_sentiment", lambda *a: None)
    monkeypatch.setattr(feature_engine, "trend_up", fake_trend)
    deps = CycleDeps(
        settings=None,
        session_factory=contextlib.nullcontext,
        brokers=[],
        kill_switch=None,
        tracker=None,
        analyst=None,
        coingecko=None,
    )
    state = {"cycle_id": "t", "cycle_started_at": NOW, "market_data": {"prices": {"AAPL": 1}}}
    features = feature_engine.compute_features(state, deps=deps)["features"]
    assert features["AAPL"]["trend_up"] == 1.0
    assert seen == [[today - timedelta(days=2), today - timedelta(days=1)]]
