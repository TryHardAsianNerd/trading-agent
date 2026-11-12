"""Alpaca market data: daily bars, crypto quotes and news, through alpaca-py.

Stocks use the free IEX feed (SIP is paid and must never be requested), split- but
not dividend-adjusted so closes match venue prices. Daily candles are stamped 00:00
UTC of their date like every other source, and crypto volume is in US dollars. SDK
errors keep only the type and HTTP status: the SDK's text can embed bodies and URLs.
"""

from collections.abc import Callable
from datetime import UTC, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.models.bars import Bar, BarSet
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestTradeRequest,
    NewsRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame

from trading_agent.data.models import AssetClass, Candle, NewsArticle, Quote
from trading_agent.data.providers.base import ProviderError, ProviderPermissionError

SOURCE = "alpaca"
_MARKET_TZ = ZoneInfo("America/New_York")


class AlpacaDataClient:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        client: StockHistoricalDataClient | None = None,
        crypto_client: CryptoHistoricalDataClient | None = None,
        news_client: NewsClient | None = None,
    ) -> None:
        """Create the stock, crypto and news clients; ValueError without keys or a stock client."""
        if client is None:
            if not api_key or not secret_key:
                raise ValueError("AlpacaDataClient requires ALPACA_API_KEY and ALPACA_SECRET_KEY")
            client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self._client = client
        self._crypto = crypto_client or CryptoHistoricalDataClient(
            api_key=api_key, secret_key=secret_key
        )
        self._news = news_client or NewsClient(api_key=api_key, secret_key=secret_key)

    def get_daily_candles(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        """A stock's daily bars in [start, end], oldest first (today's is partial until the close).

        Raises ProviderPermissionError on HTTP 403, ProviderError on any other failure.
        """
        symbol = symbol.upper()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
            adjustment=Adjustment.SPLIT,
        )
        bars = _fetch(lambda: self._client.get_stock_bars(request), symbol, symbol, "stock")
        return _candles(bars, symbol, "stock", _trading_day_utc)

    def get_crypto_daily_candles(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        """A crypto pair's daily bars in [start, end], oldest first, with venue volume in USD.

        Raises ProviderPermissionError on HTTP 403, ProviderError on any other failure.
        """
        symbol = symbol.upper()
        pair = symbol.replace("-", "/")
        request = CryptoBarsRequest(
            symbol_or_symbols=pair, timeframe=TimeFrame.Day, start=start, end=end
        )
        bars = _fetch(lambda: self._crypto.get_crypto_bars(request), pair, symbol, "crypto")
        return _candles(bars, symbol, "crypto", _utc_midnight)

    def get_quote(self, symbol: str) -> Quote:
        """The latest trade price of a crypto pair; ProviderError on any failure or no trade."""
        symbol = symbol.upper()
        pair = symbol.replace("-", "/")
        request = CryptoLatestTradeRequest(symbol_or_symbols=pair)
        try:
            trade = self._crypto.get_crypto_latest_trade(request).get(pair)
        except APIError as e:
            raise ProviderError(f"Alpaca HTTP {e.status_code} on quote for {symbol!r}") from None
        except Exception as e:
            raise ProviderError(
                f"Alpaca transport error on quote for {symbol!r}: {type(e).__name__}"
            ) from None
        if trade is None:
            raise ProviderError(f"Alpaca has no latest trade for {symbol!r}")
        return Quote(
            symbol=symbol,
            asset_class="crypto",
            price=Decimal(str(trade.price)),
            ts=trade.timestamp.astimezone(UTC),
            source=SOURCE,
        )

    def get_news(self, symbols: list[str], start: datetime, end: datetime) -> list[NewsArticle]:
        """Every page of headlines mentioning the symbols, each tagged with the ones it names.

        Raises ProviderPermissionError on HTTP 403, ProviderError on any other failure.
        """
        wanted = {s.upper().replace("-", ""): s.upper() for s in symbols}
        request = NewsRequest(symbols=",".join(wanted), start=start, end=end)
        try:
            news = self._news.get_news(request).data.get("news", [])
        except APIError as e:
            status = e.status_code
            if status == 403:
                raise ProviderPermissionError("Alpaca 403 on news - plan not permitted") from None
            raise ProviderError(f"Alpaca HTTP {status} on news") from None
        except Exception as e:
            raise ProviderError(f"Alpaca transport error on news: {type(e).__name__}") from None
        try:
            return [
                NewsArticle(
                    id=item.id,
                    created_at=item.created_at,
                    headline=item.headline,
                    source=item.source,
                    symbols=sorted({wanted[s] for s in item.symbols if s in wanted}),
                )
                for item in news
            ]
        except (AttributeError, TypeError, ValueError) as e:
            raise ProviderError(f"Alpaca news payload malformed: {type(e).__name__}") from None


def _fetch(call: Callable[[], BarSet], key: str, symbol: str, kind: str) -> list[Bar]:
    """Run one bars request and return the bars for key, with sanitized errors."""
    try:
        return call().data.get(key, [])
    except APIError as e:
        status = e.status_code
        if status == 403:
            raise ProviderPermissionError(
                f"Alpaca 403 on {kind} bars for {symbol!r} — feed/plan not permitted"
            ) from None
        if status == 429:
            raise ProviderError("Alpaca rate limit hit — caller should back off") from None
        raise ProviderError(f"Alpaca HTTP {status} on {kind} bars for {symbol!r}") from None
    except Exception as e:
        raise ProviderError(
            f"Alpaca transport error on {kind} bars for {symbol!r}: {type(e).__name__}"
        ) from None


def _candles(
    bars: list[Bar], symbol: str, asset_class: AssetClass, day: Callable[[datetime], datetime]
) -> list[Candle]:
    """Map SDK bars to daily Candles; ProviderError if any bar is malformed or mis-stamped."""
    try:
        return [
            Candle(
                symbol=symbol,
                asset_class=asset_class,
                ts=day(bar.timestamp),
                open=Decimal(str(bar.open)),
                high=Decimal(str(bar.high)),
                low=Decimal(str(bar.low)),
                close=Decimal(str(bar.close)),
                volume=Decimal(str(bar.volume))
                * (Decimal(str(bar.close)) if asset_class == "crypto" else 1),
                source=SOURCE,
                resolution="1d",
            )
            for bar in bars
        ]
    except (AttributeError, ArithmeticError, TypeError, ValueError) as e:
        raise ProviderError(f"Alpaca bar payload malformed for {symbol!r}: {e!r}") from e


def _trading_day_utc(bar_open: datetime) -> datetime:
    """Map a stock bar stamped at New York midnight to 00:00 UTC of that date.

    Any other stamp raises ValueError rather than mis-dating the bar by up to a day.
    """
    if bar_open.tzinfo is None:
        raise ValueError("Alpaca bar timestamp is timezone-naive")
    local = bar_open.astimezone(_MARKET_TZ)
    if local.time() != time(0):
        raise ValueError("Alpaca daily bar-open is not New York midnight")
    return datetime(local.year, local.month, local.day, tzinfo=UTC)


def _utc_midnight(bar_open: datetime) -> datetime:
    """Return a crypto bar's stamp in UTC; ValueError unless it is 00:00 UTC."""
    if bar_open.tzinfo is None:
        raise ValueError("Alpaca bar timestamp is timezone-naive")
    utc = bar_open.astimezone(UTC)
    if utc.time() != time(0):
        raise ValueError("Alpaca crypto daily bar-open is not 00:00 UTC")
    return utc
