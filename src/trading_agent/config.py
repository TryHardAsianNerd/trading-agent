"""Application settings, loaded from the environment and .env.

This public build is paper-only: building settings with TRADING_MODE=live raises.
"""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveTradingNotConfirmedError(RuntimeError):
    """Raised whenever TRADING_MODE=live is set: this build is paper-only."""

    def __init__(self) -> None:
        super().__init__("TRADING_MODE=live is rejected: this public build is paper-only.")


class Settings(BaseSettings):
    """App settings; secrets are repr=False so a traceback or log line never prints a key."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    trading_mode: Literal["paper", "live"] = "paper"
    live_trading_confirmed: str = ""

    # Market data
    finnhub_api_key: str = Field(default="", repr=False)
    fred_api_key: str = Field(default="", repr=False)

    # Execution
    alpaca_api_key: str = Field(default="", repr=False)
    alpaca_secret_key: str = Field(default="", repr=False)
    coinbase_api_key: str = Field(default="", repr=False)
    coinbase_api_secret: str = Field(default="", repr=False)

    # LLM analyst
    anthropic_api_key: str = Field(default="", repr=False)
    anthropic_model: str = "claude-sonnet-5"

    # Discord: alerts need token + channel; commands also need owner.
    discord_bot_token: str = Field(default="", repr=False)
    discord_channel_id: str = ""
    discord_owner_id: str = ""

    # Infrastructure
    database_url: str = Field(
        default="postgresql+psycopg://trading:trading_dev_only@127.0.0.1:5432/trading_agent",
        repr=False,  # carries the DB password
    )
    redis_url: str = "redis://127.0.0.1:6379/0"
    # Its own Redis db, apart from the kill switch.
    celery_broker_url: str = "redis://127.0.0.1:6379/1"

    # Agent behavior
    cycle_interval_minutes: int = 15
    stock_watchlist: str = "VOO,QQQM,AAPL,MSFT,NVDA,GOOGL,AMZN,META,AVGO,JPM,V,COST,WMT,LLY,XOM"
    crypto_watchlist: str = "BTC-USD,ETH-USD"
    # A string so it becomes Decimal without passing through binary float.
    paper_starting_cash: str = "10000"

    @model_validator(mode="after")
    def _enforce_live_trading_friction(self) -> "Settings":
        """Raise LiveTradingNotConfirmedError for live mode: this public build is paper-only."""
        if self.trading_mode == "live":
            raise LiveTradingNotConfirmedError()
        return self

    @property
    def is_paper(self) -> bool:
        """True when trading_mode is "paper"."""
        return self.trading_mode == "paper"

    @property
    def stock_symbols(self) -> list[str]:
        """Stock watchlist parsed into upper-cased symbols (empty entries dropped)."""
        return [s.strip().upper() for s in self.stock_watchlist.split(",") if s.strip()]

    @property
    def crypto_symbols(self) -> list[str]:
        """Crypto watchlist parsed into upper-cased symbols (empty entries dropped)."""
        return [s.strip().upper() for s in self.crypto_watchlist.split(",") if s.strip()]


def load_settings() -> Settings:
    """Load settings from the environment; raises LiveTradingNotConfirmedError like Settings."""
    return Settings()
