"""Provider clients. Each normalizes to trading_agent.data.models shapes."""

from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.data.providers.base import (
    ProviderError,
    ProviderPermissionError,
    UnknownSymbolError,
)
from trading_agent.data.providers.coingecko import CoinGeckoClient
from trading_agent.data.providers.finnhub import FinnhubClient
from trading_agent.data.providers.fred import FredClient

__all__ = [
    "AlpacaDataClient",
    "CoinGeckoClient",
    "FinnhubClient",
    "FredClient",
    "ProviderError",
    "ProviderPermissionError",
    "UnknownSymbolError",
]
