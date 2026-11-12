"""Fetch daily candles for the watchlists and upsert them.

Each symbol is fetched and persisted in its own transaction, so one bad symbol is
logged and skipped without touching the others. Stocks and crypto come from Alpaca
when a client is supplied, else from Finnhub and CoinGecko.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from trading_agent.config import Settings
from trading_agent.data.models import Candle
from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.data.providers.coingecko import MAX_DAILY_BACKFILL_DAYS, CoinGeckoClient
from trading_agent.data.providers.finnhub import FinnhubClient, ProviderPermissionError
from trading_agent.log import get_logger
from trading_agent.storage.repos import OhlcvRepository

log = get_logger(__name__)

# 365 calendar days ≈ 250 trading bars: covers the 200-bar moving averages in one request.
DEFAULT_BACKFILL_DAYS = 365


class IngestResult(dict[str, int]):
    """symbol -> number of candles upserted (absent = failed or skipped, see logs)."""


def ingest_daily_ohlcv(
    settings: Settings,
    session_factory: sessionmaker[Session],
    finnhub: FinnhubClient | None = None,
    coingecko: CoinGeckoClient | None = None,
    alpaca: AlpacaDataClient | None = None,
    backfill_days: int = DEFAULT_BACKFILL_DAYS,
    now: datetime | None = None,
) -> IngestResult:
    """Fetch and upsert daily candles for both watchlists; failed symbols are left out.

    Clients created here are closed here; clients passed in are left open.
    """
    now = now or datetime.now(UTC)
    owns_finnhub = finnhub is None and bool(settings.finnhub_api_key)
    owns_coingecko = coingecko is None and alpaca is None
    if owns_finnhub:
        finnhub = FinnhubClient(settings.finnhub_api_key)
    if owns_coingecko:
        coingecko = CoinGeckoClient()
    result = IngestResult()

    try:
        stock_source = "alpaca" if alpaca is not None else "finnhub"
        stock_client = alpaca if alpaca is not None else finnhub
        if stock_client is None:
            log.warning("ingest_stocks_skipped", reason="no ALPACA or FINNHUB keys configured")
        else:
            for symbol in settings.stock_symbols:
                candles = _fetch_stock(stock_client, stock_source, symbol, now, backfill_days)
                if candles is not None:
                    _persist(session_factory, symbol, candles, result)

        for symbol in settings.crypto_symbols:
            candles = _fetch_crypto(alpaca, coingecko, symbol, now, backfill_days)
            if candles is not None:
                _persist(session_factory, symbol, candles, result)
    finally:
        if owns_finnhub and finnhub is not None:
            finnhub.close()
        if owns_coingecko:
            coingecko.close()

    log.info("ingest_complete", upserted=dict(result))
    return result


def _fetch_stock(
    client: AlpacaDataClient | FinnhubClient,
    source: str,
    symbol: str,
    now: datetime,
    backfill_days: int,
) -> list[Candle] | None:
    """Fetch one stock's daily candles, or None after logging a failure.

    The full window is re-read every cycle so a stock split after the first backfill
    rewrites the old bars instead of skewing the ATR and moving averages.
    """
    start = now - timedelta(days=backfill_days)
    try:
        return client.get_daily_candles(symbol, start=start, end=now)
    except ProviderPermissionError:
        log.warning(
            "ingest_symbol_forbidden",
            symbol=symbol,
            provider=source,
            hint="this provider's plan does not permit daily candles",
        )
    except Exception:
        log.warning("ingest_symbol_failed", symbol=symbol, provider=source, exc_info=True)
    return None


def _fetch_crypto(
    alpaca: AlpacaDataClient | None,
    coingecko: CoinGeckoClient | None,
    symbol: str,
    now: datetime,
    backfill_days: int,
) -> list[Candle] | None:
    """Fetch one crypto symbol's daily candles, or None after logging a failure."""
    # The full window every cycle: a short window came back as 30-minute bars whose
    # first UTC day was partial and overwrote yesterday's complete bar.
    provider = "alpaca" if alpaca is not None else "coingecko"
    try:
        if alpaca is not None:
            start = now - timedelta(days=backfill_days)
            return alpaca.get_crypto_daily_candles(symbol, start=start, end=now)
        return coingecko.get_daily_candles(symbol, days=min(backfill_days, MAX_DAILY_BACKFILL_DAYS))
    except Exception:
        log.warning("ingest_symbol_failed", symbol=symbol, provider=provider, exc_info=True)
        return None


def _persist(
    session_factory: sessionmaker[Session],
    symbol: str,
    candles: list[Candle],
    result: IngestResult,
) -> None:
    """Upsert one symbol's candles in its own transaction and record the count in result."""
    try:
        with session_factory() as session:
            count = OhlcvRepository(session).upsert_candles(candles)
            session.commit()
        result[symbol] = count
    except Exception:
        log.exception("ingest_persist_failed", symbol=symbol, candles=len(candles))
