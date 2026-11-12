"""Config guardrail tests: paper mode is the default, live mode has friction."""

import pytest

from trading_agent.config import LiveTradingNotConfirmedError, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic against the ambient shell: a developer's exported TRADING_MODE
    (or any other settings var) must not leak into these tests."""
    for var in (
        "TRADING_MODE",
        "LIVE_TRADING_CONFIRMED",
        "STOCK_WATCHLIST",
        "CRYPTO_WATCHLIST",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**env: str) -> Settings:
    # _env_file=None: ignore any local .env so tests are hermetic.
    return Settings(_env_file=None, **env)


def test_default_mode_is_paper():
    s = _settings()
    assert s.trading_mode == "paper"
    assert s.is_paper


def test_live_mode_without_confirmation_raises():
    with pytest.raises(LiveTradingNotConfirmedError):
        _settings(trading_mode="live")


def test_live_mode_with_wrong_confirmation_raises():
    with pytest.raises(LiveTradingNotConfirmedError):
        _settings(trading_mode="live", live_trading_confirmed="true")


def test_live_mode_is_rejected_even_with_confirmation():
    with pytest.raises(LiveTradingNotConfirmedError, match="paper-only"):
        _settings(trading_mode="live", live_trading_confirmed="yes")


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        _settings(trading_mode="sandbox")


def test_confirmation_alone_does_not_enable_live():
    s = _settings(live_trading_confirmed="yes")
    assert s.trading_mode == "paper"


def test_watchlists_parse_and_normalize():
    s = _settings(stock_watchlist=" aapl, msft ,", crypto_watchlist="btc-usd")
    assert s.stock_symbols == ["AAPL", "MSFT"]
    assert s.crypto_symbols == ["BTC-USD"]


def test_repr_never_prints_a_credential():
    """repr/str of Settings (and of anything holding it) shows no key."""
    secret = "sk-SHOULD-NEVER-PRINT"
    settings = Settings(
        _env_file=None,
        finnhub_api_key=secret,
        fred_api_key=secret,
        alpaca_api_key=secret,
        alpaca_secret_key=secret,
        coinbase_api_key=secret,
        coinbase_api_secret=secret,
        anthropic_api_key=secret,
        discord_bot_token=secret,
        database_url=f"postgresql+psycopg://u:{secret}@h/db",
    )
    for rendered in (repr(settings), str(settings), repr([settings])):
        assert secret not in rendered
    assert settings.alpaca_api_key == secret  # still readable where it is needed
    assert "trading_mode" in repr(settings)  # non-secret fields still render
