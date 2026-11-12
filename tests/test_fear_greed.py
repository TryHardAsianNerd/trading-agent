"""Crypto Fear & Greed: the provider and its join into crypto features."""

import contextlib
from datetime import UTC, date, datetime

import httpx
import pytest
import respx
import structlog

from trading_agent.data.providers.base import ProviderError
from trading_agent.data.providers.fear_greed import URL, FearGreedClient
from trading_agent.graph.deps import CycleDeps
from trading_agent.graph.nodes import feature_engine

NOW = datetime(2026, 9, 23, 15, tzinfo=UTC)
TODAY_TS = str(int(datetime(2026, 9, 23, tzinfo=UTC).timestamp()))


def _payload(value: object = "71", ts: str = TODAY_TS) -> dict:
    """The API's JSON shape: the value and the epoch timestamp both arrive as strings."""
    return {
        "name": "Fear and Greed Index",
        "data": [{"value": value, "value_classification": "Greed", "timestamp": ts}],
        "metadata": {"error": None},
    }


@respx.mock
def test_latest_reading_and_its_utc_day():
    route = respx.get(URL).respond(json=_payload())
    assert FearGreedClient().latest() == (71, date(2026, 9, 23))
    assert route.calls.last.request.url.params["limit"] == "1"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json=_payload(value="n/a")),
        httpx.Response(200, json=_payload(value="101")),
        httpx.Response(200, json=_payload(value="-1")),
        httpx.Response(200, json=_payload(value=True)),
    ],
)
@respx.mock
def test_bad_responses_raise_provider_error(response):
    respx.get(URL).mock(return_value=response)
    with pytest.raises(ProviderError):
        FearGreedClient().latest()


@pytest.mark.parametrize("value", ["0", "100"])
@respx.mock
def test_the_scale_ends_are_valid(value):
    respx.get(URL).respond(json=_payload(value=value))
    assert FearGreedClient().latest()[0] == int(value)


@respx.mock
def test_transport_errors_carry_the_type_name_only():
    respx.get(URL).mock(side_effect=httpx.ConnectError("https://api.alternative.me/fng/?x=1"))
    with pytest.raises(ProviderError) as exc_info:
        FearGreedClient().latest()
    assert str(exc_info.value) == "Fear & Greed transport error: ConnectError"
    assert exc_info.value.__cause__ is None and exc_info.value.__suppress_context__


class _Client:
    """Returns a fixed reading, or raises."""

    def __init__(self, reading=None, error=None) -> None:
        """Hold the canned (value, day) reading or the error to raise."""
        self.reading, self.error, self.calls = reading, error, 0

    def latest(self):
        """Return the reading or raise the error."""
        self.calls += 1
        if self.error:
            raise self.error
        return self.reading


def _run(client, monkeypatch) -> tuple[dict, list]:
    """Run the feature node for one stock and one crypto symbol; return (features, log events)."""
    monkeypatch.setattr(feature_engine, "snapshot", lambda candles: {"close": 1.0})
    monkeypatch.setattr(feature_engine, "OhlcvRepository", lambda s: _NoCandles())
    monkeypatch.setattr(feature_engine, "_add_sentiment", lambda *a: None)
    deps = CycleDeps(
        settings=None,
        session_factory=contextlib.nullcontext,
        brokers=[],
        kill_switch=None,
        tracker=None,
        analyst=None,
        coingecko=None,
        fear_greed=client,
    )
    state = {
        "cycle_id": "t",
        "cycle_started_at": NOW,
        "market_data": {
            "prices": {"AAPL": 1, "BTC-USD": 1},
            "asset_classes": {"AAPL": "stock", "BTC-USD": "crypto"},
        },
    }
    with structlog.testing.capture_logs() as logs:
        features = feature_engine.compute_features(state, deps=deps)["features"]
    return features, logs


class _NoCandles:
    """Candle repository stand-in (snapshot is stubbed, so content is irrelevant)."""

    def recent_daily_candles(self, symbol, limit):
        """Return no candles."""
        return []


def test_crypto_symbols_get_the_index_and_stocks_do_not(monkeypatch):
    features, _ = _run(_Client(reading=(71, date(2026, 9, 23))), monkeypatch)
    assert features["BTC-USD"]["crypto_fear_greed"] == 71.0
    assert type(features["BTC-USD"]["crypto_fear_greed"]) is float
    assert "crypto_fear_greed" not in features["AAPL"]


def test_yesterdays_reading_is_fine_older_is_dropped(monkeypatch):
    features, _ = _run(_Client(reading=(40, date(2026, 9, 22))), monkeypatch)
    assert features["BTC-USD"]["crypto_fear_greed"] == 40.0
    features, logs = _run(_Client(reading=(40, date(2026, 9, 21))), monkeypatch)
    assert "crypto_fear_greed" not in features["BTC-USD"]
    assert [e["log_level"] for e in logs if e["event"] == "fear_greed_stale"] == ["warning"]


def test_a_future_dated_reading_is_rejected(monkeypatch):
    features, logs = _run(_Client(reading=(40, date(2026, 9, 24))), monkeypatch)
    assert "crypto_fear_greed" not in features["BTC-USD"]
    assert [e["cycle_id"] for e in logs if e["event"] == "fear_greed_stale"] == ["t"]


def test_a_failed_fetch_warns_and_keeps_everything_else(monkeypatch):
    features, logs = _run(_Client(error=ProviderError("Fear & Greed HTTP 503")), monkeypatch)
    assert features == {"AAPL": {"close": 1.0}, "BTC-USD": {"close": 1.0}}
    failed = [e for e in logs if e["event"] == "fear_greed_unavailable"]
    assert [(e["log_level"], e["reason"]) for e in failed] == [("warning", "ProviderError")]


def test_an_unexpected_client_error_is_contained(monkeypatch):
    features, logs = _run(_Client(error=KeyError("surprise")), monkeypatch)
    assert features["BTC-USD"] == {"close": 1.0}
    assert any(e["event"] == "fear_greed_unavailable" for e in logs)


def test_no_crypto_priced_means_no_call():
    client = _Client(reading=(71, date(2026, 9, 23)))
    state = {"market_data": {"asset_classes": {"AAPL": "stock"}}}
    feature_engine._add_fear_greed(client, {"AAPL": {"close": 1.0}}, state)
    assert client.calls == 0
