"""Provider exception types shared by every market-data client.

Providers must never let an httpx exception escape raw: its message embeds the
request URL, which can carry an API key.
"""


class ProviderError(RuntimeError):
    """Base error for all market-data provider failures (message is sanitized)."""


class ProviderPermissionError(ProviderError):
    """The API key/plan does not permit this endpoint."""


class UnknownSymbolError(ProviderError):
    """The provider has no mapping for the requested symbol."""
