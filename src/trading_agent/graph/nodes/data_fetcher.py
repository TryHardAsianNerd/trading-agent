"""Node 1, data fetcher: ingest daily candles, then collect current quotes and holdings.

Candles go to the database, never into the checkpointed state. A symbol whose
quote fails is left unpriced, so the gate rejects it rather than trade on a
missing price.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from trading_agent.data.ingest import ingest_daily_ohlcv
from trading_agent.graph.state import AgentState
from trading_agent.log import get_logger

if TYPE_CHECKING:
    from trading_agent.graph.deps import CycleDeps

log = get_logger(__name__)


def fetch(state: AgentState, *, deps: CycleDeps) -> dict:
    """Ingest the watchlists' daily candles and return this cycle's market_data.

    market_data holds prices and asset_classes (no stocks while the market is closed),
    ingested (rows per symbol) and held (sorted; absent when a venue's read failed).
    """
    ingested = ingest_daily_ohlcv(
        deps.settings,
        deps.session_factory,
        finnhub=deps.finnhub,
        coingecko=deps.coingecko,
        alpaca=deps.alpaca_data,
    )

    prices: dict[str, Decimal] = {}
    asset_classes: dict[str, str] = {}
    if deps.finnhub is None:
        if deps.settings.stock_symbols:
            log.warning("stock_quotes_skipped", reason="no FINNHUB_API_KEY configured")
    elif not state.get("stock_market_open", True):
        log.info("stock_quotes_skipped", reason="NYSE session closed")
    else:
        for symbol in deps.settings.stock_symbols:
            _quote(deps.finnhub, symbol, prices, asset_classes)
    # Alpaca when configured: CoinGecko's free tier refuses about a third of quote requests.
    crypto_quotes = deps.alpaca_data or deps.coingecko
    for symbol in deps.settings.crypto_symbols:
        _quote(crypto_quotes, symbol, prices, asset_classes)

    market_data: dict[str, object] = {
        "prices": prices,
        "asset_classes": asset_classes,
        "ingested": dict(ingested),
    }
    try:
        market_data["held"] = sorted(
            {p.symbol for b in deps.brokers for p in b.get_positions() if p.quantity > 0}
        )
    except Exception:
        log.warning("positions_read_failed", exc_info=True)

    log.info(
        "market_data_fetched",
        cycle_id=state.get("cycle_id"),
        priced=sorted(prices),
        held=market_data.get("held"),
        ingested=dict(ingested),
    )
    return {"market_data": market_data}


def _quote(
    provider, symbol: str, prices: dict[str, Decimal], asset_classes: dict[str, str]
) -> None:
    """Add one quote to the prices and asset_classes maps; log and skip on failure."""
    try:
        quote = provider.get_quote(symbol)
    except Exception:
        log.warning("quote_failed", symbol=symbol, exc_info=True)
        return
    prices[symbol] = quote.price
    asset_classes[symbol] = quote.asset_class
