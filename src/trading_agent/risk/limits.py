"""Risk limits as frozen constants (docs/RISK_RULES.md).

Limits are deliberately not loaded from the environment or any config file: changing
one is a reviewed code change, never a runtime setting an agent, env var or LLM output
could weaken. Where RISK_RULES.md gives a range the conservative end is used, except
where Jason chose otherwise on backtest evidence (D-032).
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Final


@dataclass(frozen=True)
class RiskLimits:
    # Layer 5 — macro pause. 30 min before: liquidity thins into the print. 60 min after:
    # the repricing, the FOMC press conference's start, and the 09:30 open after 08:30 data.
    macro_pause_before_minutes: int = 30
    macro_pause_after_minutes: int = 60
    # An older VIX close is warned about but still used: the gate only halves or doesn't,
    # so discarding it could only turn halving off (D-023).
    vix_max_age_days: int = 7

    # Layer 1 — stops. No fixed-percent trail: on daily bars it sold ordinary noise (D-031).
    atr_multiplier: Decimal = Decimal("3.0")  # stop = entry - N x ATR, N ∈ [1.5, 3.0]
    trailing_stop_pct: Decimal | None = None

    # Layer 2 — sizing
    risk_per_trade_pct: Decimal = Decimal("0.01")  # 1% of portfolio per trade (range 1-2%)
    max_position_pct: Decimal = Decimal("0.10")  # 10% cap per position (range 5-10%, D-032)
    min_order_value: Decimal = Decimal("10")  # dollars; the smallest stock order sent

    # Layer 3 — circuit breakers
    daily_loss_halt_pct: Decimal = Decimal("0.03")  # halt at 3% daily loss (range 3-5%)
    weekly_loss_pause_pct: Decimal = Decimal("0.05")  # week pause at 5% (range 5-7%)
    max_consecutive_losses: int = 5  # pause after 5 losing trades (range 5-7)
    breaker_cooldown_seconds: int = 3600  # minimum 1-hour cooldown

    # Layer 4 — kill switch
    max_drawdown_pct: Decimal = Decimal("0.15")  # kill at 15% from peak (range 15-20%)
    cash_reserve_pct: Decimal = Decimal("0.20")  # 20% cash floor, always
    # The venue fills at the quote plus slippage and fees, so sizing to 100% would be refused.
    venue_cash_usable_pct: Decimal = Decimal("0.99")

    # Layer 5 — environmental
    vix_reduce_threshold: Decimal = Decimal("30")  # halve size above this

    # Liquidity — "high-liquidity instruments only" (D-022)
    liquidity_window_days: int = 20  # ~one trading month of daily bars
    # Measured on the Alpaca IEX tape, 2.5-5% of consolidated volume: $10M IEX/day is
    # roughly $250-400M consolidated. The least liquid watchlist name (MSFT) sits ~30x
    # above it, a micro cap ~250-1000x below.
    min_avg_dollar_volume_stock: Decimal = Decimal("10000000")
    # Crypto volume is already USD. BTC/ETH trade tens of billions a day; this only
    # excludes dead markets.
    min_avg_dollar_volume_crypto: Decimal = Decimal("50000000")


DEFAULT_LIMITS: Final[RiskLimits] = RiskLimits()
