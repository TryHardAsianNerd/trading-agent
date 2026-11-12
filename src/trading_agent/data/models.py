"""Provider-agnostic market data shapes.

Prices are Decimal end to end (no float drift on money), and every timestamp must
arrive timezone-aware and is normalized to UTC.
"""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

AssetClass = Literal["stock", "crypto"]


def _require_utc(v: datetime) -> datetime:
    """Reject a naive timestamp and convert an aware one to UTC."""
    if v.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware (UTC)")
    return v.astimezone(UTC)


class Candle(BaseModel):
    symbol: str
    asset_class: AssetClass
    ts: datetime  # bar open time (UTC day start for daily bars)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source: str
    resolution: str = "1d"  # minutes ("1", "5", "60") or "1d"

    _utc = field_validator("ts")(_require_utc)


class Quote(BaseModel):
    symbol: str
    asset_class: AssetClass
    price: Decimal = Field(gt=0)  # a zero or negative quote is bad data, never a price
    ts: datetime
    source: str

    _utc = field_validator("ts")(_require_utc)


class NewsArticle(BaseModel):
    """One news headline tagged with the watchlist symbols it mentions."""

    id: int
    created_at: datetime
    headline: str
    source: str
    symbols: list[str]  # watchlist symbols only, in the repo's form ("BTC-USD")

    _utc = field_validator("created_at")(_require_utc)
