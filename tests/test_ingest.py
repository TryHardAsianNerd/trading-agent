"""Ingestion tests with fake providers and a fake session factory (no DB, no network)."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import ClassVar
from unittest.mock import MagicMock

from trading_agent.config import Settings
from trading_agent.data import ingest as ingest_mod
from trading_agent.data.ingest import ingest_daily_ohlcv
from trading_agent.data.models import Candle
from trading_agent.data.providers.finnhub import ProviderPermissionError

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _candle(symbol: str, asset_class: str, source: str) -> Candle:
    return Candle(
        symbol=symbol,
        asset_class=asset_class,
        ts=NOW,
        open=Decimal(1),
        high=Decimal(2),
        low=Decimal(1),
        close=Decimal(2),
        volume=Decimal(10),
        source=source,
    )


def _settings(**kw: str) -> Settings:
    defaults = {
        "finnhub_api_key": "k",
        "stock_watchlist": "AAPL,MSFT",
        "crypto_watchlist": "BTC-USD",
    }
    defaults.update(kw)
    return Settings(_env_file=None, **defaults)


def _session_factory() -> MagicMock:
    factory = MagicMock()
    session = MagicMock()
    factory.return_value.__enter__.return_value = session
    factory.test_session = session
    return factory


class FakeRepo:
    """Patched over ingest.OhlcvRepository; class-level state, reset per test."""

    upserted: ClassVar[list[list[Candle]]] = []
    latest: ClassVar[datetime | None] = None
    fail_symbols: ClassVar[set[str]] = set()

    def __init__(self, _session) -> None: ...

    def upsert_candles(self, candles):
        if candles and candles[0].symbol in self.fail_symbols:
            raise RuntimeError("db rejected")
        FakeRepo.upserted.append(candles)
        return len(candles)


def _use_fake_repo(monkeypatch) -> type[FakeRepo]:
    FakeRepo.upserted = []
    FakeRepo.latest = None
    FakeRepo.fail_symbols = set()
    monkeypatch.setattr(ingest_mod, "OhlcvRepository", FakeRepo)
    return FakeRepo


def _providers():
    finnhub = MagicMock()
    finnhub.get_daily_candles.side_effect = lambda s, start, end: [_candle(s, "stock", "finnhub")]
    coingecko = MagicMock()
    coingecko.get_daily_candles.side_effect = lambda s, days: [_candle(s, "crypto", "coingecko")]
    return finnhub, coingecko


def test_ingest_happy_path_commits_per_symbol(monkeypatch):
    _use_fake_repo(monkeypatch)
    finnhub, coingecko = _providers()
    factory = _session_factory()

    result = ingest_daily_ohlcv(_settings(), factory, finnhub=finnhub, coingecko=coingecko, now=NOW)

    assert result == {"AAPL": 1, "MSFT": 1, "BTC-USD": 1}
    assert len(FakeRepo.upserted) == 3
    # One transaction per persisted symbol — not one spanning the whole run.
    assert factory.test_session.commit.call_count == 3


def test_ingest_fetch_failure_does_not_starve_others(monkeypatch):
    _use_fake_repo(monkeypatch)
    finnhub, coingecko = _providers()

    def flaky(symbol, start, end):
        if symbol == "AAPL":
            raise RuntimeError("boom")
        return [_candle(symbol, "stock", "finnhub")]

    finnhub.get_daily_candles.side_effect = flaky

    result = ingest_daily_ohlcv(
        _settings(), _session_factory(), finnhub=finnhub, coingecko=coingecko, now=NOW
    )
    assert "AAPL" not in result
    assert result["MSFT"] == 1
    assert result["BTC-USD"] == 1


def test_ingest_persist_failure_does_not_starve_others(monkeypatch):
    repo = _use_fake_repo(monkeypatch)
    repo.fail_symbols = {"MSFT"}
    finnhub, coingecko = _providers()

    result = ingest_daily_ohlcv(
        _settings(), _session_factory(), finnhub=finnhub, coingecko=coingecko, now=NOW
    )
    assert "MSFT" not in result
    assert result["AAPL"] == 1
    assert result["BTC-USD"] == 1


def test_ingest_permission_error_skips_symbol(monkeypatch):
    _use_fake_repo(monkeypatch)
    finnhub, coingecko = _providers()
    finnhub.get_daily_candles.side_effect = ProviderPermissionError("403")

    result = ingest_daily_ohlcv(
        _settings(), _session_factory(), finnhub=finnhub, coingecko=coingecko, now=NOW
    )
    assert set(result) == {"BTC-USD"}


def test_ingest_without_finnhub_key_skips_stocks(monkeypatch):
    _use_fake_repo(monkeypatch)
    _, coingecko = _providers()

    result = ingest_daily_ohlcv(
        _settings(finnhub_api_key="", stock_watchlist=""),
        _session_factory(),
        coingecko=coingecko,
        now=NOW,
    )
    assert set(result) == {"BTC-USD"}


def test_ingest_crypto_always_refetches_its_whole_window(monkeypatch):
    """With stored history the crypto window is still the full 30 days: a short window
    returns 30-minute bars whose partial first day would overwrite a complete bar.
    Same two requests either way."""
    repo = _use_fake_repo(monkeypatch)
    repo.latest = NOW - timedelta(hours=3)
    finnhub, coingecko = _providers()

    ingest_daily_ohlcv(
        _settings(stock_watchlist=""),
        _session_factory(),
        finnhub=finnhub,
        coingecko=coingecko,
        now=NOW,
    )
    assert coingecko.get_daily_candles.call_args.kwargs["days"] == 30


def test_ingest_crypto_backfill_capped_at_provider_window(monkeypatch):
    """Fresh DB: crypto backfill must be capped to CoinGecko's 30-day honest-daily
    window, not the 365-day stock default."""
    _use_fake_repo(monkeypatch)
    finnhub, coingecko = _providers()

    ingest_daily_ohlcv(
        _settings(stock_watchlist=""),
        _session_factory(),
        finnhub=finnhub,
        coingecko=coingecko,
        now=NOW,
    )
    assert coingecko.get_daily_candles.call_args.kwargs["days"] == 30


def _alpaca_provider() -> MagicMock:
    alpaca = MagicMock()
    alpaca.get_daily_candles.side_effect = lambda s, start, end: [_candle(s, "stock", "alpaca")]
    alpaca.get_crypto_daily_candles.side_effect = lambda s, start, end: [
        _candle(s, "crypto", "alpaca")
    ]
    return alpaca


def test_ingest_stocks_and_crypto_use_alpaca_when_supplied(monkeypatch):
    """With an Alpaca client, stock AND crypto candles come from Alpaca (D-046):
    Finnhub's candle endpoint (its free plan 403s) and CoinGecko are never called."""
    _use_fake_repo(monkeypatch)
    finnhub, coingecko = _providers()
    alpaca = _alpaca_provider()

    result = ingest_daily_ohlcv(
        _settings(),
        _session_factory(),
        finnhub=finnhub,
        coingecko=coingecko,
        alpaca=alpaca,
        now=NOW,
    )

    assert result == {"AAPL": 1, "MSFT": 1, "BTC-USD": 1}
    finnhub.get_daily_candles.assert_not_called()
    coingecko.get_daily_candles.assert_not_called()
    assert [c.source for batch in FakeRepo.upserted for c in batch] == ["alpaca"] * 3
    for call in (alpaca.get_daily_candles, alpaca.get_crypto_daily_candles):
        kwargs = call.call_args.kwargs
        # A year of crypto bars: the core's 200-day re-entry line needs 200+.
        assert kwargs["start"] == NOW - timedelta(days=ingest_mod.DEFAULT_BACKFILL_DAYS)
        assert kwargs["end"] == NOW


def test_ingest_stocks_fall_back_to_finnhub_without_alpaca_client(monkeypatch):
    _use_fake_repo(monkeypatch)
    finnhub, coingecko = _providers()

    result = ingest_daily_ohlcv(
        _settings(), _session_factory(), finnhub=finnhub, coingecko=coingecko, now=NOW
    )

    assert result["AAPL"] == 1
    assert finnhub.get_daily_candles.call_count == 2
    assert FakeRepo.upserted[0][0].source == "finnhub"


def test_ingest_stocks_refetch_full_window_despite_stored_history(monkeypatch):
    """Stocks ignore the cursor: with history stored, the window is still the
    full backfill so a split after the initial backfill is rewritten
    (split-adjusted) on the next cycle instead of lingering in ohlcv."""
    repo = _use_fake_repo(monkeypatch)
    repo.latest = NOW - timedelta(days=3)
    finnhub, coingecko = _providers()
    alpaca = _alpaca_provider()

    ingest_daily_ohlcv(
        _settings(crypto_watchlist=""),
        _session_factory(),
        finnhub=finnhub,
        coingecko=coingecko,
        alpaca=alpaca,
        now=NOW,
    )
    kwargs = alpaca.get_daily_candles.call_args.kwargs
    assert kwargs["start"] == NOW - timedelta(days=ingest_mod.DEFAULT_BACKFILL_DAYS)
    assert kwargs["end"] == NOW


def test_ingest_alpaca_failure_isolated_per_symbol(monkeypatch):
    _use_fake_repo(monkeypatch)
    finnhub, coingecko = _providers()
    alpaca = _alpaca_provider()

    def flaky(symbol, start, end):
        if symbol == "AAPL":
            raise RuntimeError("boom")
        return [_candle(symbol, "stock", "alpaca")]

    alpaca.get_daily_candles.side_effect = flaky
    result = ingest_daily_ohlcv(
        _settings(),
        _session_factory(),
        finnhub=finnhub,
        coingecko=coingecko,
        alpaca=alpaca,
        now=NOW,
    )
    assert "AAPL" not in result
    assert result["MSFT"] == 1
    assert result["BTC-USD"] == 1


def test_ingest_closes_clients_it_created(monkeypatch):
    _use_fake_repo(monkeypatch)
    created = []

    class FakeCG:
        def __init__(self):
            created.append(self)
            self.closed = False

        def get_daily_candles(self, symbol, days):
            return [_candle(symbol, "crypto", "coingecko")]

        def close(self):
            self.closed = True

    monkeypatch.setattr(ingest_mod, "CoinGeckoClient", FakeCG)
    ingest_daily_ohlcv(
        _settings(finnhub_api_key="", stock_watchlist=""), _session_factory(), now=NOW
    )
    assert len(created) == 1
    assert created[0].closed
