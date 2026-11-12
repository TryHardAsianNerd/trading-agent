"""Finnhub (stocks): quotes and daily candles. The free plan allows 60 requests a minute.

The API key travels in the X-Finnhub-Token header, never the URL, and errors carry
only the path and status, so no logged exception can contain it.
"""

from datetime import UTC, datetime
from decimal import Decimal

import httpx

from trading_agent.data.models import Candle, Quote
from trading_agent.data.providers.base import ProviderError, ProviderPermissionError

BASE_URL = "https://finnhub.io/api/v1"


class FinnhubClient:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        """Create a Finnhub client, with a default HTTP client; ValueError without a key."""
        if not api_key:
            raise ValueError("FinnhubClient requires an API key (FINNHUB_API_KEY)")
        self._client = client or httpx.Client(timeout=10.0)
        self._headers = {"X-Finnhub-Token": api_key}

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def get_quote(self, symbol: str) -> Quote:
        """The current quote for a stock; ProviderError on any failure or a missing quote."""
        data = self._get("/quote", {"symbol": symbol})
        try:
            if not data.get("t"):
                raise ProviderError(f"Finnhub returned no quote for {symbol!r}")
            return Quote(
                symbol=symbol.upper(),
                asset_class="stock",
                price=Decimal(str(data["c"])),
                ts=datetime.fromtimestamp(data["t"], tz=UTC),
                source="finnhub",
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(f"Finnhub quote payload malformed for {symbol!r}: {e!r}") from e

    def get_daily_candles(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        """A stock's daily candles in [start, end]; empty when Finnhub has no data.

        Raises ProviderPermissionError when the plan lacks /stock/candle, ProviderError otherwise.
        """
        data = self._get(
            "/stock/candle",
            {
                "symbol": symbol,
                "resolution": "D",
                "from": int(start.timestamp()),
                "to": int(end.timestamp()),
            },
        )
        if data.get("s") == "no_data":
            return []
        if data.get("s") != "ok":
            raise ProviderError(f"Finnhub candle response not ok for {symbol!r}: {data.get('s')}")
        try:
            return [
                Candle(
                    symbol=symbol.upper(),
                    asset_class="stock",
                    ts=datetime.fromtimestamp(t, tz=UTC),
                    open=Decimal(str(o)),
                    high=Decimal(str(h)),
                    low=Decimal(str(lo)),
                    close=Decimal(str(c)),
                    volume=Decimal(str(v)),
                    source="finnhub",
                    resolution="1d",
                )
                for t, o, h, lo, c, v in zip(
                    data["t"], data["o"], data["h"], data["l"], data["c"], data["v"], strict=True
                )
            ]
        except (KeyError, TypeError, ValueError) as e:
            raise ProviderError(f"Finnhub candle payload malformed for {symbol!r}: {e!r}") from e

    def _get(self, path: str, params: dict[str, object]) -> dict:
        """GET an endpoint and return its JSON body; sanitized ProviderError on any failure."""
        try:
            resp = self._client.get(f"{BASE_URL}{path}", params=params, headers=self._headers)
        except httpx.HTTPError as e:
            raise ProviderError(f"Finnhub transport error on {path}: {type(e).__name__}") from None
        if resp.status_code == 403:
            raise ProviderPermissionError(
                f"Finnhub 403 on {path} — endpoint not available on this plan"
            )
        if resp.status_code == 429:
            raise ProviderError("Finnhub rate limit hit (60/min) — caller should back off")
        if resp.status_code >= 400:
            raise ProviderError(f"Finnhub HTTP {resp.status_code} on {path}")
        return resp.json()
