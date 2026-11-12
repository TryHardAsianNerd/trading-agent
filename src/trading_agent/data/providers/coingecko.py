"""CoinGecko (free tier, no key): crypto spot quotes and daily OHLCV.

The free /ohlc endpoint never returns daily bars (30-minute at 1 day, 4-hour up to
30 days, 4-day beyond), so daily candles are built from intraday bars, capping
history at 30 days. Daily volume is market_chart's trailing-24h figure, an
approximation of the day's traded volume.
"""

import time
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx

from trading_agent.data.models import Candle, Quote
from trading_agent.data.providers.base import ProviderError, UnknownSymbolError

BASE_URL = "https://api.coingecko.com/api/v3"

# /ohlc's accepted values, restricted to windows returning intraday bars (90+ gives 4-day bars).
_OHLC_DAYS_CHOICES = (1, 7, 14, 30)
MAX_DAILY_BACKFILL_DAYS = 30
# One /simple/price call prices every mapped coin for this long: the free tier 429s
# back-to-back calls, and a cycle asks for each coin within seconds.
QUOTE_TTL_SECONDS = 30.0

_COIN_IDS: dict[str, str] = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "DOGE-USD": "dogecoin",
    "ADA-USD": "cardano",
}


def _coin_id(symbol: str) -> str:
    """Map a watchlist symbol ("BTC-USD") to its CoinGecko id; UnknownSymbolError if unmapped."""
    try:
        return _COIN_IDS[symbol.upper()]
    except KeyError:
        raise UnknownSymbolError(
            f"No CoinGecko id mapped for {symbol!r} — add it to _COIN_IDS"
        ) from None


def snap_days(days: int) -> int:
    """Smallest /ohlc-accepted window covering days, capped at 30."""
    for choice in _OHLC_DAYS_CHOICES:
        if days <= choice:
            return choice
    return MAX_DAILY_BACKFILL_DAYS


# A volume point stamped exactly at 00:00 belongs to the day that just ended.
_EPSILON = timedelta(seconds=1)


class CoinGeckoClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        """Create a CoinGecko client, with a default HTTP client when none is given."""
        self._client = client or httpx.Client(timeout=10.0)
        self._prices: dict[str, Any] = {}
        self._priced_at = float("-inf")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def get_quote(self, symbol: str) -> Quote:
        """The current USD price of a crypto symbol, from one cached call for every coin."""
        coin = _coin_id(symbol)
        if time.monotonic() - self._priced_at > QUOTE_TTL_SECONDS:
            ids = ",".join(sorted(set(_COIN_IDS.values())))
            self._prices = self._get(
                "/simple/price",
                {"ids": ids, "vs_currencies": "usd", "include_last_updated_at": "true"},
            )
            self._priced_at = time.monotonic()
        data = self._prices
        try:
            return Quote(
                symbol=symbol.upper(),
                asset_class="crypto",
                price=Decimal(str(data[coin]["usd"])),
                ts=datetime.fromtimestamp(data[coin]["last_updated_at"], tz=UTC),
                source="coingecko",
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(f"CoinGecko price payload malformed for {symbol!r}: {e!r}") from e

    def get_daily_candles(self, symbol: str, days: int = MAX_DAILY_BACKFILL_DAYS) -> list[Candle]:
        """Daily candles built from intraday bars, at most 30 days back.

        Left out: a first day that starts mid-day (partial open, high and low), and any
        day whose volume point is not published yet, so the stored row is kept.
        """
        coin = _coin_id(symbol)
        window = snap_days(days)
        bar = _bar_length(window)
        ohlc = self._get(f"/coins/{coin}/ohlc", {"vs_currency": "usd", "days": str(window)})
        chart = self._get(
            f"/coins/{coin}/market_chart",
            {"vs_currency": "usd", "days": str(window), "interval": "daily"},
        )
        try:
            # Each point is the trailing 24h volume at its stamp: the 00:00 point of D+1
            # is day D's volume, and the latest point stands in for today.
            volumes_by_day: dict[date, Decimal] = {
                (datetime.fromtimestamp(ms / 1000, tz=UTC) - _EPSILON).date(): Decimal(str(vol))
                for ms, vol in chart.get("total_volumes", [])
            }
            buckets: dict[date, list[tuple[datetime, Decimal, Decimal, Decimal, Decimal]]] = (
                defaultdict(list)
            )
            for ms, o, h, lo, c in ohlc:
                # CoinGecko stamps a bar at its close: bucket by the bar's open so day D
                # runs 00:00 -> 24:00 UTC.
                ts = datetime.fromtimestamp(ms / 1000, tz=UTC) - bar
                buckets[ts.date()].append(
                    (ts, Decimal(str(o)), Decimal(str(h)), Decimal(str(lo)), Decimal(str(c)))
                )
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(f"CoinGecko OHLC payload malformed for {symbol!r}: {e!r}") from e

        candles = []
        first_day = min(buckets, default=None)
        for day in sorted(buckets):
            bars = sorted(buckets[day], key=lambda b: b[0])
            if day == first_day and bars[0][0].time() != datetime.min.time():
                continue
            if day not in volumes_by_day:
                continue
            candles.append(
                Candle(
                    symbol=symbol.upper(),
                    asset_class="crypto",
                    ts=datetime(day.year, day.month, day.day, tzinfo=UTC),
                    open=bars[0][1],
                    high=max(b[2] for b in bars),
                    low=min(b[3] for b in bars),
                    close=bars[-1][4],
                    volume=volumes_by_day[day],
                    source="coingecko",
                    resolution="1d",
                )
            )
        return candles

    def _get(self, path: str, params: dict[str, str]) -> dict | list:
        """GET an endpoint and return its JSON body; ProviderError on any failure (sanitized)."""
        try:
            resp = self._client.get(f"{BASE_URL}{path}", params=params)
        except httpx.HTTPError as e:
            raise ProviderError(
                f"CoinGecko transport error on {path}: {type(e).__name__}"
            ) from None
        if resp.status_code == 429:
            raise ProviderError("CoinGecko rate limit hit (~30/min) — caller should back off")
        if resp.status_code >= 400:
            raise ProviderError(f"CoinGecko HTTP {resp.status_code} on {path}")
        return resp.json()


def _bar_length(window: int) -> timedelta:
    """Bar length the free /ohlc returns for a window: 30 minutes up to 2 days, else 4 hours."""
    return timedelta(minutes=30) if window <= 2 else timedelta(hours=4)
