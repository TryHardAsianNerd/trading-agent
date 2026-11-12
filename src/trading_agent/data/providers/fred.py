"""FRED client: the latest VIX daily close (series VIXCLS) for the risk gate.

The close is normally a trading day old. The gate only halves buys above 30, so a
stale close is still used; the risk node warns about it.

FRED takes its API key as a URL query parameter, so every request URL is a secret.
Errors here carry only a type name or status, with the chain cut. Separately, httpx
logs the full URL on success: configure_logging pins its loggers to WARNING and
log.redact_url_secrets scrubs key parameters. Both are tested; keep both.
"""

from datetime import date
from decimal import Decimal

import httpx

from trading_agent.data.providers.base import ProviderError

BASE_URL = "https://api.stlouisfed.org/fred"
SERIES_ID = "VIXCLS"
# Enough recent rows to skip weekend/holiday "." placeholders and still find a close.
OBSERVATION_LIMIT = 10
# This endpoint takes 6-10s; a 10s timeout failed more often than not.
TIMEOUT_SECONDS = 20.0


class FredClient:
    def __init__(self, api_key: str, client: httpx.Client | None = None) -> None:
        """Create a FRED client, with a default HTTP client; ValueError without a key."""
        if not api_key:
            raise ValueError("FredClient requires an API key (FRED_API_KEY)")
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=TIMEOUT_SECONDS)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def latest_vix(self) -> tuple[Decimal, date]:
        """The latest published VIX close and its date; ProviderError on any failure or no close."""
        params = {
            "series_id": SERIES_ID,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": OBSERVATION_LIMIT,
        }
        try:
            resp = self._client.get(f"{BASE_URL}/series/observations", params=params)
        except Exception as e:
            # Broader than httpx.HTTPError on purpose: anything raised with the
            # key-bearing URL in hand leaves with its type name only.
            raise ProviderError(f"FRED transport error: {type(e).__name__}") from None
        if resp.status_code >= 400:
            raise ProviderError(f"FRED HTTP {resp.status_code}")
        try:
            for obs in resp.json()["observations"]:
                if obs["value"] != ".":
                    return Decimal(obs["value"]), date.fromisoformat(obs["date"])
        except Exception as e:  # json/Decimal/Key errors can quote the payload: type name only
            raise ProviderError(f"FRED payload malformed: {type(e).__name__}") from None
        raise ProviderError(
            f"FRED returned no {SERIES_ID} value in the last {OBSERVATION_LIMIT} observations"
        )
