"""Provider client tests with mocked HTTP (respx). No network.

CoinGecko mocks mirror the REAL free-tier API shape:
/ohlc returns intraday bars (30m/4h), never daily; the client aggregates.
"""

import contextlib
import io
import logging
import traceback
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import respx
import structlog
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.models.bars import BarSet
from alpaca.data.timeframe import TimeFrame
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError

from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.data.providers.base import (
    ProviderError,
    ProviderPermissionError,
    UnknownSymbolError,
)
from trading_agent.data.providers.coingecko import (
    BASE_URL as CG_BASE,
)
from trading_agent.data.providers.coingecko import (
    CoinGeckoClient,
    snap_days,
)
from trading_agent.data.providers.finnhub import (
    BASE_URL as FH_BASE,
)
from trading_agent.data.providers.finnhub import (
    FinnhubClient,
)
from trading_agent.data.providers.fred import (
    BASE_URL as FRED_BASE,
)
from trading_agent.data.providers.fred import (
    FredClient,
)
from trading_agent.log import _scrub, configure_logging, drain_alerts, get_logger

SECRET = "SECRET_KEY_VALUE"


# --- Finnhub ---------------------------------------------------------------


@respx.mock
def test_finnhub_quote_parses_and_authenticates_via_header():
    route = respx.get(f"{FH_BASE}/quote").respond(json={"c": 232.47, "t": 1756857600})
    q = FinnhubClient(SECRET).get_quote("aapl")
    assert q.symbol == "AAPL"
    assert q.price == Decimal("232.47")
    assert q.ts == datetime.fromtimestamp(1756857600, tz=UTC)
    request = route.calls.last.request
    assert request.headers["X-Finnhub-Token"] == SECRET
    assert SECRET not in str(request.url)  # key never travels in the URL


@respx.mock
def test_finnhub_candles_parse_and_align():
    respx.get(f"{FH_BASE}/stock/candle").respond(
        json={
            "s": "ok",
            "t": [1756771200, 1756857600],
            "o": [230.0, 232.0],
            "h": [233.5, 234.1],
            "l": [229.9, 231.0],
            "c": [232.4, 233.9],
            "v": [1000, 2000],
        }
    )
    candles = FinnhubClient(SECRET).get_daily_candles(
        "AAPL", start=datetime(2026, 9, 1, tzinfo=UTC), end=datetime(2026, 9, 3, tzinfo=UTC)
    )
    assert len(candles) == 2
    assert candles[0].close == Decimal("232.4")
    assert candles[1].volume == Decimal("2000")
    assert all(c.resolution == "1d" and c.asset_class == "stock" for c in candles)


@respx.mock
def test_finnhub_no_data_returns_empty():
    respx.get(f"{FH_BASE}/stock/candle").respond(json={"s": "no_data"})
    assert (
        FinnhubClient(SECRET).get_daily_candles(
            "AAPL", start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 2, tzinfo=UTC)
        )
        == []
    )


@respx.mock
def test_finnhub_403_raises_permission_error():
    respx.get(f"{FH_BASE}/stock/candle").respond(status_code=403)
    with pytest.raises(ProviderPermissionError):
        FinnhubClient(SECRET).get_daily_candles(
            "AAPL", start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 1, 2, tzinfo=UTC)
        )


@pytest.mark.parametrize("status", [401, 429, 500])
@respx.mock
def test_finnhub_errors_never_contain_the_key(status):
    """Guardrail: no exception from any HTTP failure may carry the API key,
    because ingest logs exceptions and logs must never contain secrets."""
    respx.get(f"{FH_BASE}/quote").respond(status_code=status)
    with pytest.raises(ProviderError) as exc_info:
        FinnhubClient(SECRET).get_quote("AAPL")
    assert SECRET not in str(exc_info.value)
    assert SECRET not in repr(exc_info.value)


@respx.mock
def test_finnhub_malformed_payload_raises_provider_error_without_key():
    respx.get(f"{FH_BASE}/quote").respond(json={"t": 1756857600})  # missing "c"
    with pytest.raises(ProviderError) as exc_info:
        FinnhubClient(SECRET).get_quote("AAPL")
    assert SECRET not in str(exc_info.value)


def test_finnhub_requires_key():
    with pytest.raises(ValueError):
        FinnhubClient("")


# --- CoinGecko -------------------------------------------------------------

DAY_MS = 86_400_000
D0 = 1756771200000  # 2025-09-02 00:00 UTC


def _four_hour_bars(day_start_ms: int, opens: list[float]) -> list[list[float]]:
    """4h bars for one UTC day, STAMPED AT THEIR CLOSE like the real API:
    the day's first bar (00:00-04:00) carries 04:00, its last carries the next
    day's 00:00."""
    bars = []
    for i, o in enumerate(opens):
        ts = day_start_ms + (i + 1) * 4 * 3_600_000
        bars.append([ts, o, o + 5, o - 5, o + 1])
    return bars


@respx.mock
def test_coingecko_quote_parses():
    respx.get(f"{CG_BASE}/simple/price").respond(
        json={"bitcoin": {"usd": 109253.12, "last_updated_at": 1756857600}}
    )
    q = CoinGeckoClient().get_quote("BTC-USD")
    assert q.symbol == "BTC-USD"
    assert q.price == Decimal("109253.12")
    assert q.asset_class == "crypto"


@respx.mock
def test_coingecko_prices_every_coin_in_one_call_and_reuses_it(monkeypatch):
    """The free tier 429s back-to-back calls: a cycle's quotes and its paper
    fills must share one request per QUOTE_TTL_SECONDS."""
    from trading_agent.data.providers import coingecko

    route = respx.get(f"{CG_BASE}/simple/price").respond(
        json={
            "bitcoin": {"usd": 100.0, "last_updated_at": 1756857600},
            "ethereum": {"usd": 10.0, "last_updated_at": 1756857600},
        }
    )
    clock = [1000.0]
    monkeypatch.setattr(coingecko.time, "monotonic", lambda: clock[0])
    client = CoinGeckoClient()
    assert client.get_quote("BTC-USD").price == Decimal(100)
    assert client.get_quote("ETH-USD").price == Decimal(10)
    assert client.get_quote("BTC-USD").price == Decimal(100)
    assert route.call_count == 1
    assert {"bitcoin", "ethereum"} <= set(route.calls[0].request.url.params["ids"].split(","))
    clock[0] += coingecko.QUOTE_TTL_SECONDS + 1
    client.get_quote("BTC-USD")
    assert route.call_count == 2


@respx.mock
def test_coingecko_aggregates_intraday_bars_into_daily():
    day1 = _four_hour_bars(D0, [100, 102, 104, 103, 101, 105])
    day2 = _four_hour_bars(D0 + DAY_MS, [106, 108, 107, 109, 110, 111])
    respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=day1 + day2)
    # Trailing-24h volume: the point at D+1 00:00 is day D's.
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(
        json={"total_volumes": [[D0 + DAY_MS, 111.0], [D0 + 2 * DAY_MS, 222.0]]}
    )
    candles = CoinGeckoClient().get_daily_candles("BTC-USD", days=7)

    assert len(candles) == 2
    d1 = candles[0]
    assert d1.ts == datetime(2025, 9, 2, tzinfo=UTC)  # bucket labeled at UTC day start
    assert d1.open == Decimal("100")  # first bar's open
    assert d1.high == Decimal("110")  # max(o+5) = 105+5
    assert d1.low == Decimal("95")  # min(o-5) = 100-5
    assert d1.close == Decimal("106")  # last bar's close = 105+1
    assert d1.volume == Decimal("111.0")
    assert d1.resolution == "1d"
    d2 = candles[1]
    assert d2.open == Decimal("106")
    assert d2.close == Decimal("112")
    assert d2.volume == Decimal("222.0")


@respx.mock
def test_coingecko_day_is_midnight_to_midnight_not_20_to_20():
    """A bar stamped 00:00 closes the PREVIOUS day, so the 20:00-24:00 bar stays in
    its own day."""
    last_bar_of_day1 = [D0 + DAY_MS, 1.0, 999.0, 1.0, 555.0]  # 20:00-24:00 on day 1
    ohlc = [*_four_hour_bars(D0, [100] * 5), last_bar_of_day1]
    respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=ohlc)
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(
        json={"total_volumes": [[D0 + DAY_MS, 1.0]]}
    )
    (day1,) = CoinGeckoClient().get_daily_candles("BTC-USD", days=7)
    assert day1.ts == datetime(2025, 9, 2, tzinfo=UTC)
    assert (day1.high, day1.close) == (Decimal("999.0"), Decimal("555.0"))


@respx.mock
def test_coingecko_drops_a_first_day_that_starts_mid_day():
    """The window starts at 'now - N days', mid-day: that day's open/high/low
    are partial and must not overwrite a stored complete bar."""
    partial = _four_hour_bars(D0, [100, 102, 104, 103, 101, 105])[3:]  # from 12:00
    full = _four_hour_bars(D0 + DAY_MS, [106, 108, 107, 109, 110, 111])
    respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=partial + full)
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(
        json={"total_volumes": [[D0 + DAY_MS, 1.0], [D0 + 2 * DAY_MS, 2.0]]}
    )
    candles = CoinGeckoClient().get_daily_candles("BTC-USD", days=7)
    assert [c.ts for c in candles] == [datetime(2025, 9, 3, tzinfo=UTC)]


@respx.mock
def test_coingecko_one_day_window_uses_30_minute_bars():
    """days=1 returns 30-minute bars: a bar stamped 00:30 opened at 00:00."""
    ohlc = [[D0 + (i + 1) * 1_800_000, 10.0 + i, 11.0 + i, 9.0 + i, 10.5 + i] for i in range(48)]
    respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=ohlc)
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(
        json={"total_volumes": [[D0 + DAY_MS, 1.0]]}
    )
    (day,) = CoinGeckoClient().get_daily_candles("BTC-USD", days=1)
    assert (day.ts, day.open, day.close) == (datetime(2025, 9, 2, tzinfo=UTC), 10, Decimal("57.5"))


@respx.mock
def test_coingecko_requests_only_enumerated_days_values():
    """days must be snapped to an API-accepted value or the real API 400s."""
    route_ohlc = respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=[])
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(json={"total_volumes": []})
    CoinGeckoClient().get_daily_candles("BTC-USD", days=3)  # 3 is NOT accepted by the API
    assert route_ohlc.calls.last.request.url.params["days"] == "7"


def test_snap_days_covers_and_caps():
    assert snap_days(1) == 1
    assert snap_days(3) == 7
    assert snap_days(10) == 14
    assert snap_days(30) == 30
    assert snap_days(365) == 30  # hard cap: beyond 30 the API returns 4-day bars


@respx.mock
def test_coingecko_day_without_a_volume_point_is_left_out():
    """Just after midnight yesterday's 00:00 volume point may not be published:
    writing volume 0 would overwrite a good stored row, so the day is skipped."""
    ohlc = [*_four_hour_bars(D0, [1, 2, 3, 4, 5, 6]), *_four_hour_bars(D0 + DAY_MS, [7, 8])]
    respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=ohlc)
    # Only a "now" point on day 2 - day 1's D+1 00:00 point is missing.
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(
        json={"total_volumes": [[D0 + DAY_MS + 9 * 3_600_000, 7.0]]}
    )
    candles = CoinGeckoClient().get_daily_candles("BTC-USD", days=7)
    assert [(c.ts.day, c.volume) for c in candles] == [(3, Decimal("7.0"))]


@respx.mock
def test_coingecko_now_volume_point_goes_to_today_beside_a_midnight_point():
    ohlc = [*_four_hour_bars(D0, [1, 2, 3, 4, 5, 6]), *_four_hour_bars(D0 + DAY_MS, [7])]
    respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=ohlc)
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(
        json={"total_volumes": [[D0 + DAY_MS, 111.0], [D0 + DAY_MS + 5 * 3_600_000, 22.0]]}
    )
    day1, day2 = CoinGeckoClient().get_daily_candles("BTC-USD", days=7)
    assert (day1.volume, day2.volume) == (Decimal("111.0"), Decimal("22.0"))


@respx.mock
def test_coingecko_a_lone_bar_before_the_first_midnight_is_dropped():
    """A 4h window's first bar can be stamped 00:00 - the 20:00-24:00 bar of
    the day before: a one-bar partial day that must not be stored."""
    ohlc = [[D0, 9.0, 9.0, 9.0, 9.0], *_four_hour_bars(D0, [1, 2, 3, 4, 5, 6])]
    respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=ohlc)
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(
        json={"total_volumes": [[D0, 1.0], [D0 + DAY_MS, 2.0]]}
    )
    (day,) = CoinGeckoClient().get_daily_candles("BTC-USD", days=7)
    assert (day.ts, day.open) == (datetime(2025, 9, 2, tzinfo=UTC), Decimal("1"))


@respx.mock
def test_coingecko_malformed_payload_raises_provider_error():
    respx.get(f"{CG_BASE}/coins/bitcoin/ohlc").respond(json=[[D0, 1.0]])  # wrong arity
    respx.get(f"{CG_BASE}/coins/bitcoin/market_chart").respond(json={"total_volumes": []})
    with pytest.raises(ProviderError):
        CoinGeckoClient().get_daily_candles("BTC-USD", days=1)


def test_coingecko_unknown_symbol():
    with pytest.raises(UnknownSymbolError):
        CoinGeckoClient().get_quote("XYZ-USD")


def test_provider_errors_share_a_base():
    """Callers must be able to `except ProviderError` once across providers."""
    from trading_agent.data.providers import alpaca, coingecko, finnhub

    assert finnhub.ProviderError is coingecko.ProviderError is ProviderError
    assert alpaca.ProviderError is ProviderError


# --- Alpaca (mocked SDK) ---------------------------------------------------


def _raw_bar(t: str, o: float, h: float, lo: float, c: float, v: float) -> dict:
    """One bar in the shape the alpaca-py BarSet parser expects."""
    return {"t": t, "o": o, "h": h, "l": lo, "c": c, "v": v, "n": 10, "vw": o}


def _alpaca(client: MagicMock) -> AlpacaDataClient:
    return AlpacaDataClient(SECRET, SECRET, client=client)


WINDOW = {"start": datetime(2026, 1, 1, tzinfo=UTC), "end": datetime(2026, 7, 10, tzinfo=UTC)}


def test_alpaca_bars_map_to_candles_at_utc_day_start():
    """Alpaca stamps daily bars at NY midnight (05:00 UTC in winter, 04:00 in
    summer); Candle.ts must be 00:00 UTC of that trading date, matching the
    repo's daily-candle convention so dedupe-by-timestamp stays exact."""
    client = MagicMock()
    client.get_stock_bars.return_value = BarSet(
        {
            "AAPL": [
                _raw_bar("2026-01-05T05:00:00Z", 230.0, 233.5, 229.9, 232.4, 1000),
                _raw_bar("2026-07-06T04:00:00Z", 232.0, 234.1, 231.0, 233.9, 2000),
            ]
        }
    )
    candles = _alpaca(client).get_daily_candles("aapl", **WINDOW)
    assert [c.ts for c in candles] == [
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 7, 6, tzinfo=UTC),
    ]
    assert candles[0].symbol == "AAPL"
    assert candles[0].open == Decimal("230.0")
    assert candles[0].close == Decimal("232.4")
    assert candles[1].volume == Decimal("2000")
    assert all(
        c.source == "alpaca" and c.asset_class == "stock" and c.resolution == "1d" for c in candles
    )


def test_alpaca_requests_iex_feed_daily_split_adjusted():
    """Guardrail: the free data plan is IEX; SIP is paid and must never be
    requested. Daily timeframe, split-adjusted (never dividend-adjusted, which
    would move closes away from venue prices), and exactly the window asked for."""
    client = MagicMock()
    client.get_stock_bars.return_value = BarSet({})
    _alpaca(client).get_daily_candles("AAPL", **WINDOW)
    request = client.get_stock_bars.call_args.args[0]
    assert request.feed == DataFeed.IEX
    assert request.adjustment == Adjustment.SPLIT
    assert request.timeframe.value == TimeFrame.Day.value
    assert request.symbol_or_symbols == "AAPL"
    # The SDK normalizes aware inputs to naive UTC (astimezone(utc) then strip).
    assert request.start == WINDOW["start"].replace(tzinfo=None)
    assert request.end == WINDOW["end"].replace(tzinfo=None)


def test_alpaca_empty_response_returns_empty_list():
    client = MagicMock()
    client.get_stock_bars.return_value = BarSet({})
    assert _alpaca(client).get_daily_candles("AAPL", **WINDOW) == []


def _api_error(status: int, body: str) -> APIError:
    http_error = HTTPError(response=MagicMock(status_code=status))
    return APIError(body, http_error)


def test_alpaca_403_raises_permission_error_without_body():
    body = f'{{"code": 40310000, "message": "subscription does not permit {SECRET}"}}'
    client = MagicMock()
    client.get_stock_bars.side_effect = _api_error(403, body)
    with pytest.raises(ProviderPermissionError) as exc_info:
        _alpaca(client).get_daily_candles("AAPL", **WINDOW)
    rendered = _rendered(exc_info.value)
    assert SECRET not in rendered
    assert "subscription" not in rendered


def _rendered(exc: BaseException) -> str:
    """The traceback exactly as structlog renders it when ingest logs exc_info=True:
    includes __cause__ AND __context__ chains, so a plain `raise` inside the
    except block (instead of `from None`) would surface the SDK text here."""
    return "".join(traceback.format_exception(exc))


@pytest.mark.parametrize("status", [401, 429, 500])
def test_alpaca_api_errors_carry_status_only(status):
    """Guardrail: alpaca-py's APIError text is the raw response body; ingest
    logs exceptions, so only the status may propagate."""
    client = MagicMock()
    client.get_stock_bars.side_effect = _api_error(status, f'{{"message": "{SECRET}"}}')
    with pytest.raises(ProviderError) as exc_info:
        _alpaca(client).get_daily_candles("AAPL", **WINDOW)
    assert not isinstance(exc_info.value, ProviderPermissionError)
    assert SECRET not in _rendered(exc_info.value)


def test_alpaca_transport_errors_never_contain_the_url():
    client = MagicMock()
    client.get_stock_bars.side_effect = RequestsConnectionError(
        f"https://data.alpaca.markets/v2/stocks/bars?key={SECRET}"
    )
    with pytest.raises(ProviderError) as exc_info:
        _alpaca(client).get_daily_candles("AAPL", **WINDOW)
    rendered = _rendered(exc_info.value)
    assert SECRET not in rendered
    assert "alpaca.markets" not in rendered
    assert "ConnectionError" in str(exc_info.value)


def test_alpaca_malformed_bar_raises_provider_error():
    bar = MagicMock()
    bar.timestamp = datetime(2026, 1, 5, 5, tzinfo=UTC)
    bar.open = None  # Decimal("None") is an InvalidOperation, not a ValueError
    client = MagicMock()
    client.get_stock_bars.return_value.data = {"AAPL": [bar]}
    with pytest.raises(ProviderError):
        _alpaca(client).get_daily_candles("AAPL", **WINDOW)


@pytest.mark.parametrize(
    "stamp",
    [
        datetime(2026, 1, 5, 5),  # naive: would be silently mis-dated
        datetime(2026, 3, 9, 0, tzinfo=UTC),  # UTC midnight = 19:00 NY the day before
        datetime(2026, 1, 5, 14, 30, tzinfo=UTC),  # an intraday bar-open
    ],
)
def test_alpaca_bar_open_must_be_new_york_midnight(stamp):
    """The normalizer maps NY-midnight stamps only; anything else (a changed
    stamping convention, a non-daily timeframe) fails loudly instead of
    shifting every bar by a day."""
    bar = MagicMock()
    bar.timestamp = stamp
    bar.open = bar.high = bar.low = bar.close = bar.volume = 1.0
    client = MagicMock()
    client.get_stock_bars.return_value.data = {"AAPL": [bar]}
    with pytest.raises(ProviderError):
        _alpaca(client).get_daily_candles("AAPL", **WINDOW)


def test_alpaca_requires_keys():
    with pytest.raises(ValueError):
        AlpacaDataClient("", SECRET)
    with pytest.raises(ValueError):
        AlpacaDataClient(SECRET, "")


def _alpaca_crypto(crypto: MagicMock) -> AlpacaDataClient:
    return AlpacaDataClient(SECRET, SECRET, client=MagicMock(), crypto_client=crypto)


def test_alpaca_crypto_bars_map_to_repo_symbols_at_utc_midnight():
    """Training history (D-024): the watchlist's BTC-USD is Alpaca's BTC/USD;
    candles come back under the repo's symbol, already at 00:00 UTC."""
    crypto = MagicMock()
    crypto.get_crypto_bars.return_value = BarSet(
        {
            "BTC/USD": [
                _raw_bar("2026-01-05T00:00:00Z", 90000, 91000, 89000, 90500, 1.5),
                _raw_bar("2026-01-06T00:00:00Z", 90500, 92000, 90000, 91800, 2.5),
            ]
        }
    )
    candles = _alpaca_crypto(crypto).get_crypto_daily_candles("btc-usd", **WINDOW)
    request = crypto.get_crypto_bars.call_args.args[0]
    assert request.symbol_or_symbols == "BTC/USD"
    assert request.timeframe.value == TimeFrame.Day.value
    assert [c.ts for c in candles] == [
        datetime(2026, 1, 5, tzinfo=UTC),
        datetime(2026, 1, 6, tzinfo=UTC),
    ]
    assert {(c.symbol, c.asset_class, c.source) for c in candles} == {
        ("BTC-USD", "crypto", "alpaca")
    }
    assert candles[1].close == Decimal("91800")
    # Crypto volume is USD everywhere (D-046): 2.5 BTC x 91,800.
    assert candles[1].volume == Decimal("2.5") * Decimal("91800")


def test_alpaca_crypto_quote_is_the_latest_trade():
    """D-046: the cycle's crypto price comes from Alpaca's latest trade."""
    crypto = MagicMock()
    at = datetime(2026, 9, 30, 6, 45, tzinfo=UTC)
    crypto.get_crypto_latest_trade.return_value = {
        "BTC/USD": SimpleNamespace(price=82984.17, timestamp=at)
    }
    quote = _alpaca_crypto(crypto).get_quote("btc-usd")
    assert crypto.get_crypto_latest_trade.call_args.args[0].symbol_or_symbols == "BTC/USD"
    assert (quote.symbol, quote.asset_class, quote.source) == ("BTC-USD", "crypto", "alpaca")
    assert (quote.price, quote.ts) == (Decimal("82984.17"), at)


def test_alpaca_crypto_quote_failures_are_sanitized_provider_errors():
    crypto = MagicMock()
    crypto.get_crypto_latest_trade.return_value = {}
    with pytest.raises(ProviderError, match="no latest trade"):
        _alpaca_crypto(crypto).get_quote("BTC-USD")
    crypto.get_crypto_latest_trade.side_effect = _api_error(500, f'{{"message": "{SECRET}"}}')
    with pytest.raises(ProviderError) as exc_info:
        _alpaca_crypto(crypto).get_quote("BTC-USD")
    assert SECRET not in _rendered(exc_info.value)


def test_alpaca_crypto_bar_open_must_be_utc_midnight():
    bar = MagicMock()
    bar.timestamp = datetime(2026, 1, 5, 5, tzinfo=UTC)
    bar.open = bar.high = bar.low = bar.close = bar.volume = 1.0
    crypto = MagicMock()
    crypto.get_crypto_bars.return_value.data = {"BTC/USD": [bar]}
    with pytest.raises(ProviderError):
        _alpaca_crypto(crypto).get_crypto_daily_candles("BTC-USD", **WINDOW)


def test_alpaca_crypto_errors_are_sanitized():
    crypto = MagicMock()
    crypto.get_crypto_bars.side_effect = _api_error(500, f'{{"message": "{SECRET}"}}')
    with pytest.raises(ProviderError) as exc_info:
        _alpaca_crypto(crypto).get_crypto_daily_candles("BTC-USD", **WINDOW)
    assert SECRET not in _rendered(exc_info.value)
    assert "crypto" in str(exc_info.value)


# --- FRED ------------------------------------------------------------------
#
# FRED is the one provider whose API key travels in the URL, so its tests are
# mostly containment: the key must reach the request and nothing else.

OBS_URL = f"{FRED_BASE}/series/observations"


def _observations(*rows: tuple[str, str]) -> dict:
    """A FRED observations payload, newest first; "." is FRED's missing-value marker."""
    return {"observations": [{"date": d, "value": v} for d, v in rows]}


@respx.mock
def test_fred_returns_the_latest_close_and_its_date():
    route = respx.get(OBS_URL).respond(json=_observations(("2026-09-21", "17.43")))
    vix, on = FredClient(SECRET).latest_vix()
    assert vix == Decimal("17.43")
    assert isinstance(vix, Decimal)  # never a float: risk inputs stay exact
    assert on == date(2026, 9, 21)
    params = route.calls.last.request.url.params
    assert params["series_id"] == "VIXCLS"
    assert params["sort_order"] == "desc"
    assert params["api_key"] == SECRET  # FRED offers no header auth


@respx.mock
def test_fred_skips_missing_values_and_takes_the_first_real_close():
    """Weekends and holidays are published as "." rows; the newest REAL close is
    the one the gate uses, and the date returned is that close's date."""
    respx.get(OBS_URL).respond(
        json=_observations(("2026-09-21", "."), ("2026-09-20", "."), ("2026-09-18", "16.02"))
    )
    vix, on = FredClient(SECRET).latest_vix()
    assert vix == Decimal("16.02")
    assert on == date(2026, 9, 18)


@respx.mock
def test_fred_raises_when_every_observation_is_missing():
    respx.get(OBS_URL).respond(json=_observations(("2026-09-21", "."), ("2026-09-20", ".")))
    with pytest.raises(ProviderError):
        FredClient(SECRET).latest_vix()


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
@respx.mock
def test_fred_http_errors_carry_status_only_and_never_the_key(status):
    """A 400 from FRED is usually a bad api_key, and the body echoes it back."""
    respx.get(OBS_URL).respond(status_code=status, text=f"Bad Request. api_key={SECRET}")
    with pytest.raises(ProviderError) as exc_info:
        FredClient(SECRET).latest_vix()
    rendered = _rendered(exc_info.value)
    assert SECRET not in rendered
    assert "stlouisfed" not in rendered
    assert str(status) in str(exc_info.value)


@respx.mock
def test_fred_transport_errors_never_contain_the_url():
    """httpx embeds the full request URL in ConnectError text, and that URL IS
    the secret. `from None` must cut the chain: format_exception renders
    __cause__ AND __context__, which is what structlog logs on exc_info."""
    respx.get(OBS_URL).mock(
        side_effect=httpx.ConnectError(f"failed to connect to {OBS_URL}?api_key={SECRET}")
    )
    with pytest.raises(ProviderError) as exc_info:
        FredClient(SECRET).latest_vix()
    rendered = _rendered(exc_info.value)
    assert SECRET not in rendered
    assert "stlouisfed" not in rendered
    assert "ConnectError" in str(exc_info.value)


@respx.mock
def test_fred_malformed_json_is_sanitized():
    respx.get(OBS_URL).respond(text=f"<html>error: api_key={SECRET}</html>")
    with pytest.raises(ProviderError) as exc_info:
        FredClient(SECRET).latest_vix()
    assert SECRET not in _rendered(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"error_message": "bad key"},  # no "observations" at all
        {"observations": [{"date": "2026-09-21"}]},  # no "value"
        {"observations": [{"date": "2026-09-21", "value": "not-a-number"}]},
        {"observations": [{"date": "nonsense", "value": "17.4"}]},
    ],
)
@respx.mock
def test_fred_malformed_payloads_raise_provider_error(payload):
    respx.get(OBS_URL).respond(json=payload)
    with pytest.raises(ProviderError) as exc_info:
        FredClient(SECRET).latest_vix()
    assert SECRET not in _rendered(exc_info.value)


@contextlib.contextmanager
def _captured_log_stream():
    """Configure real logging and yield the StringIO the root handler renders JSON into.

    Root handlers and the httpx/httpcore levels are restored on exit: configure_logging
    clears handlers process-wide, which would otherwise leak into later tests.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_levels = {n: logging.getLogger(n).level for n in ("httpx", "httpcore")}
    stream = io.StringIO()
    configure_logging()
    handler = logging.StreamHandler(stream)
    # The SAME ProcessorFormatter the real stdout handler uses — a bare handler would
    # render the raw message and test nothing about the production pipeline. Found by
    # type, not by index: pytest's logging plugin also installs a root handler.
    handler.setFormatter(
        next(
            h.formatter
            for h in root.handlers
            if isinstance(h.formatter, structlog.stdlib.ProcessorFormatter)
        )
    )
    root.addHandler(handler)
    try:
        yield stream
    finally:
        root.handlers = saved_handlers
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)


@respx.mock
def test_fred_key_never_reaches_the_log_stream():
    """The success path is the leak sanitized exceptions do NOT cover: httpx
    logs `GET <full url>` at INFO and configure_logging bridges third-party
    records into the same JSON stream.

    NOTE: this passes if EITHER defence holds (the level pin or the redaction
    processor), so it does not guard the pin on its own — that is
    test_configure_logging_pins_the_noisy_http_loggers' job. It is the
    end-to-end statement that a real fetch leaks nothing."""
    respx.get(OBS_URL).respond(json=_observations(("2026-09-21", "17.43")))
    with _captured_log_stream() as stream:
        FredClient(SECRET).latest_vix()
        # Belt and braces: even an explicit httpx record at INFO stays unrendered.
        logging.getLogger("httpx").info("HTTP Request: GET %s?api_key=%s", OBS_URL, SECRET)
    assert SECRET not in stream.getvalue()


@pytest.mark.parametrize("logger_name", ["httpx", "httpcore.http11", "some_future_library"])
def test_url_secrets_are_redacted_whatever_the_logger_or_level(logger_name):
    """The level pin only silences two names at one level, so it must not BE
    the whole defence: a library promoting its request line to WARNING, or any
    other logger quoting a key-bearing URL, would leak. `redact_url_secrets`
    scrubs at the render choke point instead. Without it, a routine httpx
    upgrade could write the live FRED key to stdout every 15 minutes."""
    with _captured_log_stream() as stream:
        logging.getLogger(logger_name).warning("GET %s?api_key=%s&x=1", OBS_URL, SECRET)
    rendered = stream.getvalue()
    assert SECRET not in rendered
    assert "api_key=REDACTED" in rendered


def test_configure_logging_pins_the_noisy_http_loggers():
    """Layer 1 on its own. Without this, deleting the pin breaks nothing red:
    the end-to-end test above still passes because the redaction processor
    covers the same line, so the two layers silently collapse into one."""
    with _captured_log_stream():
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING


@pytest.mark.parametrize(
    "param",
    [
        "api_key",
        "apikey",
        "API_KEY",
        "apiKey",
        "api-key",
        "x-api-key",
        "x_cg_pro_api_key",  # CoinGecko Pro: an existing provider, one plan upgrade away
        "token",  # Tiingo
        "access_token",
        "key",
        "secret",
        "auth",
        "sig",
        "signature",
    ],
)
def test_any_credential_shaped_query_param_is_redacted(param):
    """The param list is the thing most likely to be out of date when the next
    key-in-URL provider lands, so it matches on the SHAPE of the name."""
    with _captured_log_stream() as stream:
        logging.getLogger("any_library").warning("GET https://x/y?%s=%s&z=1", param, SECRET)
    rendered = stream.getvalue()
    assert SECRET not in rendered
    assert f"{param}=REDACTED" in rendered


@pytest.mark.parametrize("word", ["monkey", "turkey", "token_id", "sort_order"])
def test_ordinary_fields_are_not_mangled_by_the_redactor(word):
    """Every trade decision must stay reconstructable from logs, so the regex
    must not eat values whose names merely END in a credential word."""
    with _captured_log_stream() as stream:
        get_logger("test").warning("evt", detail=f"{word}=12345")
    assert f"{word}=12345" in stream.getvalue()


def test_redaction_descends_into_containers_and_objects():
    """The leak this layer exists for is the one nobody anticipated. A URL can
    reach a log field inside a dict, a list, or an httpx.Request repr —
    `log.warning("failed", request=exc.request)` is one line away in any future
    provider, and JSONRenderer emits containers natively and repr()s the rest."""
    url = f"{OBS_URL}?api_key={SECRET}&file_type=json"
    with _captured_log_stream() as stream:
        get_logger("test").warning(
            "provider_failed",
            nested={"outer": {"url": url}},
            listed=[url],
            request=httpx.Request("GET", url),
            raw=url.encode(),
            keyed={url: 1},  # a URL is as likely to be a dict KEY as a value
        )
    rendered = stream.getvalue()
    assert SECRET not in rendered
    assert rendered.count("REDACTED") >= 5


class _Exploding:
    """A value whose repr raises — the redactor must not propagate that."""

    def __repr__(self):
        raise RuntimeError("repr blew up")


def test_a_log_call_never_raises_into_the_caller():
    """Logging is best-effort (D-018): a log line must never take down the
    executor or the scheduler. The redactor runs in structlog's chain on the
    CALLER's thread, so unlike a RENDER failure (which `logging` swallows in
    the handler) anything it raises escapes into application code."""

    for field in [
        {("AAPL", "alpaca"): 3},  # tuple key: scrubbing it would return an unhashable list
        {frozenset({"a"}): 1},
        {1: 2},
        _Exploding(),
    ]:
        with _captured_log_stream():
            get_logger("test").warning("evt", m=field)  # must not raise


def test_scrub_renders_non_string_dict_keys_as_strings():
    """Pins the tuple-key conversion itself, not the safety net (the processor's
    try/except would mask a failure by replacing the whole field). A tuple key
    becomes its str(), so json.dumps can render it; a str() that carries a
    secret is scrubbed like any other key."""
    scrubbed = _scrub(
        {("AAPL", "alpaca"): 3, 1: 2, f"?api_key={SECRET}": 4, (f"u?token={SECRET}",): 5}
    )
    assert scrubbed["('AAPL', 'alpaca')"] == 3
    assert scrubbed["1"] == 2
    assert scrubbed["?api_key=REDACTED"] == 4
    assert SECRET not in repr(scrubbed)


def test_a_non_string_key_never_drops_the_whole_record():
    """End to end: a tuple-keyed dict renders, so the record keeps equity and cycle_id
    instead of json.dumps failing inside the handler."""
    with _captured_log_stream() as stream:
        get_logger("t").warning(
            "cycle_counters", m={("AAPL", "alpaca"): 3}, equity="100000", cycle_id="abc"
        )
    rendered = stream.getvalue()
    assert '"equity": "100000"' in rendered and '"cycle_id": "abc"' in rendered
    assert "('AAPL', 'alpaca')" in rendered


def test_an_unscrubbable_value_is_dropped_not_passed_through():
    """Fail CLOSED: the value the redactor could not process is exactly the one
    that might hold a secret, so it is replaced rather than let through."""
    with _captured_log_stream() as stream:
        get_logger("test").warning("evt", boom=_Exploding(), fine="ok")
    rendered = stream.getvalue()
    assert "redaction failed: RuntimeError" in rendered
    assert '"fine": "ok"' in rendered  # the rest of the event still survives


def test_redaction_happens_before_alerts_are_collected_for_discord():
    """Pins the PROCESSOR ORDER, not just the rendered output. `collect_alerts`
    copies scalar fields into the Discord buffer at the moment it runs, so if it
    ever ran before `redact_url_secrets` the alert would carry the key while the
    stdout JSON still looked clean — a leak invisible in the place you would
    look for it. D-018's "an alert field that can hold library text is a policy
    violation", in its key-in-URL form."""
    drain_alerts()  # start from an empty buffer; it is a module-level global
    with _captured_log_stream() as stream:
        get_logger("test").error("venue_call_failed", detail=f"{OBS_URL}?api_key={SECRET}")
    alerts = drain_alerts()
    assert SECRET not in stream.getvalue()
    assert alerts, "the error-level event should have been buffered for Discord"
    assert SECRET not in str(alerts)
    assert "api_key=REDACTED" in alerts[0]["detail"]


def test_redaction_also_covers_our_own_structlog_events_and_tracebacks():
    """Our own fields and the rendered traceback go through the same scrub —
    the traceback only becomes a string in the final processor chain."""
    with _captured_log_stream() as stream:
        log = get_logger("test")
        log.warning("manual", detail=f"{OBS_URL}?api_key={SECRET}")
        try:
            raise RuntimeError(f"boom {OBS_URL}?token={SECRET}")
        except RuntimeError:
            log.exception("exploded")
    rendered = stream.getvalue()
    assert SECRET not in rendered
    assert "REDACTED" in rendered


def test_fred_requires_a_key():
    with pytest.raises(ValueError):
        FredClient("")


def _alpaca_news(news: MagicMock) -> AlpacaDataClient:
    """A data client whose news SDK is the given mock."""
    return AlpacaDataClient(SECRET, SECRET, client=MagicMock(), news_client=news)


def _news_item(i: int, symbols: list[str]) -> MagicMock:
    """One SDK news item tagged with the given Alpaca symbols."""
    item = MagicMock()
    item.id = i
    item.created_at = datetime(2026, 9, 22, 20, tzinfo=UTC)
    item.headline = f"headline {i}"
    item.source = "benzinga"
    item.symbols = symbols
    return item


def test_alpaca_news_maps_symbols_both_ways_and_keeps_only_the_watchlist():
    """Crypto is requested as BTCUSD and comes back as the repo's BTC-USD;
    tickers outside the watchlist are dropped from each article."""
    news = MagicMock()
    news.get_news.return_value.data = {
        "news": [_news_item(1, ["BTCUSD", "ZECUSD"]), _news_item(2, ["AAPL", "MSFT", "NVDA"])]
    }
    articles = _alpaca_news(news).get_news(["AAPL", "btc-usd", "NVDA"], **WINDOW)
    request = news.get_news.call_args.args[0]
    assert request.symbols == "AAPL,BTCUSD,NVDA"
    assert [(a.id, a.symbols) for a in articles] == [(1, ["BTC-USD"]), (2, ["AAPL", "NVDA"])]


def test_alpaca_news_errors_are_sanitized():
    news = MagicMock()
    news.get_news.side_effect = _api_error(500, f'{{"message": "{SECRET}"}}')
    with pytest.raises(ProviderError) as exc_info:
        _alpaca_news(news).get_news(["AAPL"], **WINDOW)
    assert SECRET not in _rendered(exc_info.value)
    news.get_news.side_effect = RequestsConnectionError(f"https://x/?key={SECRET}")
    with pytest.raises(ProviderError) as exc_info:
        _alpaca_news(news).get_news(["AAPL"], **WINDOW)
    assert SECRET not in _rendered(exc_info.value)


@pytest.mark.parametrize("price", ["0", "-1"])
def test_a_quote_is_never_zero_or_negative(price):
    """A bad quote must fail at entry, not read as a price through a stop and sell."""
    from pydantic import ValidationError

    from trading_agent.data.models import Quote

    with pytest.raises(ValidationError):
        Quote(
            symbol="AAPL",
            asset_class="stock",
            price=Decimal(price),
            ts=datetime.now(UTC),
            source="t",
        )
