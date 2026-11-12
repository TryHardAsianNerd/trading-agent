"""Synthetic daily candles for the ML tests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np

from trading_agent.data.models import Candle

START = datetime(2022, 1, 3, tzinfo=UTC)


def candles_from_closes(
    closes: list[float], symbol: str = "AAA", asset_class: str = "stock", start: datetime = START
) -> list[Candle]:
    """One ascending daily candle per close from `start`, high/low 1% around it."""
    return [
        Candle(
            symbol=symbol,
            asset_class=asset_class,
            ts=start + timedelta(days=i),
            open=Decimal(str(c)),
            high=Decimal(str(round(c * 1.01, 6))),
            low=Decimal(str(round(c * 0.99, 6))),
            close=Decimal(str(c)),
            volume=Decimal("1000"),
            source="itest",
        )
        for i, c in enumerate(closes)
    ]


def random_walk(n: int, seed: int, trend_block: int | None = None) -> list[float]:
    """A positive price path of `n` closes, rounded to 4 dp.

    With `trend_block` set, the drift flips sign every `trend_block` bars (+/-1%/day)
    under 1%/day noise — a planted, learnable momentum signal. None is pure noise.
    """
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    if trend_block:
        drift = np.where((np.arange(n) // trend_block) % 2 == 0, 0.01, -0.01)
    returns = drift + rng.normal(0, 0.01, n)
    return [round(float(p), 4) for p in 100 * np.exp(np.cumsum(returns))]
