"""Crypto Fear & Greed Index from alternative.me: a free daily 0-100 market sentiment gauge."""

from datetime import UTC, date, datetime

import httpx

from trading_agent.data.providers.base import ProviderError

URL = "https://api.alternative.me/fng/"
TIMEOUT_SECONDS = 10.0


class FearGreedClient:
    def __init__(self, client: httpx.Client | None = None) -> None:
        """Create a Fear & Greed client, with a default HTTP client when none is given."""
        self._client = client or httpx.Client(timeout=TIMEOUT_SECONDS)

    def latest(self) -> tuple[int, date]:
        """The latest index value (0-100) and its UTC date; ProviderError on any failure."""
        try:
            resp = self._client.get(URL, params={"limit": 1, "format": "json"})
        except Exception as e:
            raise ProviderError(f"Fear & Greed transport error: {type(e).__name__}") from None
        if resp.status_code >= 400:
            raise ProviderError(f"Fear & Greed HTTP {resp.status_code}")
        try:
            row = resp.json()["data"][0]
            if isinstance(row["value"], bool):  # int(True) == 1 would pass the range check
                raise TypeError("boolean value")
            value = int(row["value"])
            day = datetime.fromtimestamp(int(row["timestamp"]), tz=UTC).date()
        except Exception as e:
            raise ProviderError(f"Fear & Greed payload malformed: {type(e).__name__}") from None
        if not 0 <= value <= 100:
            raise ProviderError("Fear & Greed value outside 0-100")
        return value, day
