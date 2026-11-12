"""The data fetcher: crypto quote source (D-046) and the account's holdings."""

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trading_agent.brokers.base import BrokerPosition
from trading_agent.data.models import Quote
from trading_agent.graph.nodes import data_fetcher


def _provider(price: str) -> MagicMock:
    """A quote provider returning one fixed crypto price."""
    provider = MagicMock()
    provider.get_quote.side_effect = lambda s: Quote(
        symbol=s, asset_class="crypto", price=Decimal(price), ts=datetime.now(UTC), source="t"
    )
    return provider


@pytest.mark.parametrize(("with_alpaca", "expected"), [(True, "2"), (False, "1")])
def test_crypto_quotes_come_from_alpaca_when_configured(monkeypatch, with_alpaca, expected):
    monkeypatch.setattr(data_fetcher, "ingest_daily_ohlcv", lambda *a, **k: {})
    coingecko, alpaca = _provider("1"), _provider("2")
    deps = SimpleNamespace(
        settings=SimpleNamespace(stock_symbols=[], crypto_symbols=["BTC-USD"]),
        session_factory=None,
        finnhub=None,
        coingecko=coingecko,
        alpaca_data=alpaca if with_alpaca else None,
        brokers=[],
    )

    out = data_fetcher.fetch({"cycle_id": "c"}, deps=deps)

    assert out["market_data"]["prices"] == {"BTC-USD": Decimal(expected)}
    assert coingecko.get_quote.called is not with_alpaca


def _broker(*symbols: str) -> MagicMock:
    """A venue holding the given symbols."""
    broker = MagicMock()
    broker.get_positions.return_value = [
        BrokerPosition(symbol=s, asset_class="stock", quantity=Decimal(1)) for s in symbols
    ]
    return broker


def _fetch(monkeypatch, brokers: list) -> dict:
    """Run the fetch step (candle ingestion stubbed) for one crypto symbol; return market_data."""
    monkeypatch.setattr(data_fetcher, "ingest_daily_ohlcv", lambda *a, **k: {})
    deps = SimpleNamespace(
        settings=SimpleNamespace(stock_symbols=[], crypto_symbols=["BTC-USD"]),
        session_factory=None,
        finnhub=None,
        coingecko=_provider("1"),
        alpaca_data=None,
        brokers=brokers,
    )
    return data_fetcher.fetch({"cycle_id": "c"}, deps=deps)["market_data"]


def test_held_lists_every_venues_positions(monkeypatch):
    """Positions from every venue, sorted; a zero-quantity row is not held (as at the gate)."""
    stocks = _broker("MSFT", "AAPL")
    stocks.get_positions.return_value.append(
        BrokerPosition(symbol="JPM", asset_class="stock", quantity=Decimal(0))
    )
    market = _fetch(monkeypatch, [stocks, _broker("BTC-USD")])
    assert market["held"] == ["AAPL", "BTC-USD", "MSFT"]


def test_a_failed_position_read_leaves_held_out_and_keeps_the_quotes(monkeypatch):
    """One venue down: holdings unknown, but the quotes are still collected."""
    down = _broker()
    down.get_positions.side_effect = RuntimeError("venue down")
    market = _fetch(monkeypatch, [_broker("AAPL"), down])
    assert "held" not in market
    assert market["prices"] == {"BTC-USD": Decimal("1")}
