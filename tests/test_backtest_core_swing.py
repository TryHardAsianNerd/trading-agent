"""Core + swing backtest with deposits: metrics, core rules, the simulated book."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from tests.ml_fakes import candles_from_closes

from trading_agent.backtest import core_swing as cs
from trading_agent.backtest.core_swing import CoreStop, _CoreHolding, core_buys
from trading_agent.backtest.engine import backtest_limits
from trading_agent.backtest.metrics import (
    deposit_and_hold,
    money_weighted_return,
    time_weighted_drawdown,
)
from trading_agent.data.models import Candle

T0 = datetime(2023, 1, 1, tzinfo=UTC)
WARM = 210  # bars before the simulation starts: the 200-day line exists from day one
D = Decimal


def _series(closes: list[float], symbol: str, asset_class: str = "stock") -> list[Candle]:
    """Liquid daily candles from T0, one per close, on every calendar day."""
    return [
        c.model_copy(update={"volume": D(10**9)})
        for c in candles_from_closes(closes, symbol, asset_class, T0)
    ]


def _core(voo: list[float], n: int, other: float = 100.0) -> dict[str, list[Candle]]:
    """Core history for every CORE_TARGETS symbol: VOO from `voo`, the rest flat at `other`."""
    return {
        "VOO": _series(voo, "VOO"),
        "QQQM": _series([other] * n, "QQQM"),
        "BTC-USD": _series([other] * n, "BTC-USD", "crypto"),
        "ETH-USD": _series([other] * n, "ETH-USD", "crypto"),
    }


def _day(i: int) -> date:
    """The i-th day from T0."""
    return (T0 + timedelta(days=i)).date()


def _spy_allocation(monkeypatch) -> list[set[str]]:
    """Record the eligible set of every core_buys call in the returned list."""
    calls: list[set[str]] = []
    real = cs.core_buys

    def spy(stock_claim, venue_cash, values, eligible, equity):
        """Record the eligible set, then plan the buys as usual."""
        calls.append(set(eligible))
        return real(stock_claim, venue_cash, values, eligible, equity)

    monkeypatch.setattr(cs, "core_buys", spy)
    return calls


# --- metrics ---


def test_money_weighted_return_known_values():
    d0 = date(2024, 1, 1)
    assert money_weighted_return([(d0, 1000.0)], d0 + timedelta(days=365), 1100.0) == 0.1
    # 100 compounding two years + 100 one year, both at 10%: 121 + 110.
    flows = [(d0, 100.0), (d0 + timedelta(days=365), 100.0)]
    assert money_weighted_return(flows, d0 + timedelta(days=730), 231.0) == 0.1
    assert money_weighted_return([], d0, 0.0) == 0.0


def test_a_deposit_never_hides_a_fall_in_the_drawdown():
    """The raw balance makes a new high on the deposit day; the strategy lost 40%."""
    days = pd.date_range("2024-01-01", periods=3, freq="D")
    equity = pd.Series([100.0, 80.0, 560.0], index=days)
    deposits = pd.Series({days[2]: 500.0})
    assert time_weighted_drawdown(equity, deposits) == 0.4
    assert time_weighted_drawdown(equity.iloc[:1], deposits) == 0.0


def test_deposit_and_hold_buys_the_open_after_the_deposit_day():
    bars = [
        c.model_copy(update={"open": D(p)})
        for c, p in zip(
            candles_from_closes([100, 200, 200], "VOO", "stock", T0), [90, 100, 200], strict=True
        )
    ]
    curve = deposit_and_hold(bars, {_day(0): 100.0}, _day(2), D("0.01"))
    assert curve.iloc[0] == 100.0  # cash on its own day, never day 0's open
    assert curve.iloc[1] == pytest.approx(100 / 1.01 * 2)  # bought at day 1's open of 100
    assert curve.iloc[2] == pytest.approx(100 / 1.01 * 2)


# --- core rules ---


def _first_deposit_buys(eligible: set[str]) -> dict[str, Decimal]:
    """The core's buys in dollars from the first $1,000 on an empty account."""
    venues = {"stock": D(900), "crypto": D(100)}
    return core_buys(D(600), venues, {}, eligible, D(1000))


def test_each_venue_buys_its_own_core_symbols_from_its_own_cash():
    alloc = _first_deposit_buys(set(cs.CORE_TARGETS))
    usable = D("0.99")
    assert alloc["BTC-USD"] + alloc["ETH-USD"] == pytest.approx(D(100) * usable)
    assert alloc["VOO"] + alloc["QQQM"] == pytest.approx(D(600) * usable)
    assert alloc["VOO"] / alloc["QQQM"] == pytest.approx(D("0.375") / D("0.225"))


def test_a_stopped_out_holdings_share_stays_in_cash():
    alloc = _first_deposit_buys(set(cs.CORE_TARGETS) - {"VOO"})
    assert "VOO" not in alloc
    assert alloc["QQQM"] == pytest.approx(D(600) * D("0.99") * D("0.375"))


def test_core_stop_rules():
    h = _CoreHolding(quantity=D(1), avg_cost=D(100), atr=D(2), high_close=D(120))
    assert CoreStop(atr=D(3)).breached(D(94), h, None)
    assert not CoreStop(atr=D(3)).breached(D("94.01"), h, None)
    assert CoreStop(trail=D("0.25")).breached(D(90), h, None)
    assert not CoreStop(trail=D("0.25")).breached(D("90.01"), h, None)
    assert CoreStop(below_trend=True).breached(D(99), h, D(100))
    assert not CoreStop(below_trend=True).breached(D(99), h, None)
    assert not CoreStop().breached(D(1), h, D(100))


def test_first_day_waits_for_every_lookback_and_refuses_short_history():
    history = _core([100.0] * 250, 250)
    history["AAPL"] = _series([10.0] * 250, "AAPL")
    assert cs.first_day(history) == _day(200)
    history["QQQM"] = history["QQQM"][:150]
    with pytest.raises(ValueError, match="QQQM"):
        cs.first_day(history)


# --- the simulated book ---


def test_deposits_land_every_14_days_and_are_never_profit():
    n = WARM + 60
    history = _core([100.0] * n, n)
    result = cs.simulate_book(
        history, _day(WARM), _day(n - 1), CoreStop(), "everything", backtest_limits()
    )
    assert list(result.deposits) == [_day(WARM + 14 * i) for i in range(5)]
    assert list(result.deposits.values()) == [D(1000)] + [D(500)] * 4
    summary = cs.summarize_book(result)
    # Flat prices: the only change is what the fills cost; nothing is profit.
    assert summary["deposited"] == 3000
    assert -3000 * 0.008 < summary["profit"] < 0
    assert summary["kill_switch"] is None
    assert result.equity.iloc[0] == 1000  # the deposit day, before any fill


def test_a_breached_core_stop_sells_and_waits_for_the_200_day_line(monkeypatch):
    calls = _spy_allocation(monkeypatch)
    # VOO: flat, a 30% fall (the 15% trail sells), a stretch below the line, then above it.
    voo = [100.0] * (WARM + 20) + [70.0] * 30 + [130.0] * 30
    history = _core(voo, len(voo))
    result = cs.simulate_book(
        history,
        _day(WARM),
        _day(len(voo) - 1),
        CoreStop(trail=D("0.15")),
        "swing only",
        backtest_limits(),
    )
    assert result.core_exits == 1
    with_voo = ["VOO" in c for c in calls]
    first_out = with_voo.index(False)
    back_in = with_voo.index(True, first_out)
    assert first_out == 20  # the close at 70 (day 20 of the run) sells and bars the buys
    assert all(not v for v in with_voo[first_out:back_in])
    assert back_in == 50  # the first close at 130, above the 200-day average


def test_a_stop_above_the_line_waits_for_a_cross_back_above_it(monkeypatch):
    """A 16% one-day drop far above the 200-day average sells on the 15% trail; the
    holding stays out while the price sits above the line and returns only once it
    has closed below the line and then above it - never a two-day round trip."""
    calls = _spy_allocation(monkeypatch)
    rising = [100.0 + i for i in range(WARM + 20)]
    voo = rising + [rising[-1] * 0.84] * 30 + [150.0] * 10 + [400.0] * 10
    history = _core(voo, len(voo))
    result = cs.simulate_book(
        history,
        _day(WARM),
        _day(len(voo) - 1),
        CoreStop(trail=D("0.15")),
        "swing only",
        backtest_limits(),
    )
    assert result.core_exits == 1
    with_voo = ["VOO" in c for c in calls]
    first_out = with_voo.index(False)
    assert first_out == 20
    assert with_voo.index(True, first_out) == 60  # the first close at 400


def test_a_loss_on_a_deposit_day_still_trips_the_daily_breaker():
    """The deposit must not hide the day's loss: the day-start anchor includes it."""
    n = WARM + 20
    fall = [100.0] * (WARM + 14) + [90.0] * 6  # the whole core falls 10% on the 2nd deposit day
    history = {
        s: _series(fall, s, "crypto" if s.endswith("-USD") else "stock") for s in cs.CORE_TARGETS
    }
    # A minimum order no buy reaches: the gate sees a buy proposal every day.
    history["MSFT"] = _series([50_000 + 50 * i for i in range(n)], "MSFT")
    limits = replace(backtest_limits(), min_order_value=Decimal(10**9))
    result = cs.simulate_book(history, _day(WARM), _day(n - 1), CoreStop(), "everything", limits)
    assert _day(WARM + 14) in result.deposits
    assert result.rejections.get("daily_loss_breaker") == 1
    assert result.halted_on is None


def test_the_kill_switch_scope_decides_whether_core_buying_continues(monkeypatch):
    n = WARM + 90
    crash = [100.0] * (WARM + 30) + [50.0] * 60  # the whole core halves: drawdown far past 15%
    results = {}
    for scope in cs.KILL_SCOPES:
        calls = _spy_allocation(monkeypatch)
        history = _core(crash, n, other=100.0)
        for s in ("QQQM", "BTC-USD", "ETH-USD"):
            history[s] = _series(crash, s, history[s][0].asset_class)
        result = cs.simulate_book(
            history, _day(WARM), _day(n - 1), CoreStop(), scope, backtest_limits()
        )
        results[scope] = (result.halted_on, len(calls))
    halted, everything_calls = results["everything"]
    assert halted == _day(WARM + 30)
    assert everything_calls == 30  # none from the close that armed the switch onward
    assert results["swing only"] == (halted, 90)


def test_swing_buys_are_fractional_even_when_no_share_fits():
    n = WARM + 160
    history = _core([100.0] * n, n)
    # Both trend up (buy), then down (sell); no whole MSFT share fits the sleeve.
    up_down = [10 + 0.05 * i for i in range(WARM + 100)] + [15.0 - 0.1 * i for i in range(60)]
    history["AAPL"] = _series(up_down, "AAPL")
    history["MSFT"] = _series([5_000 * c for c in up_down], "MSFT")
    result = cs.simulate_book(
        history, _day(WARM), _day(n - 1), CoreStop(), "everything", backtest_limits()
    )
    assert {t.symbol for t in result.swing_trades} == {"AAPL", "MSFT"}
    assert all(t.quantity < 1 for t in result.swing_trades if t.symbol == "MSFT")
    summary = cs.summarize_book(result)
    assert summary["swing_trades"] == len(result.swing_trades)
    # The cap is 25% of the SWING sleeve's equity, not of the account.
    first = result.swing_trades[0]
    sleeve = result.swing_equity[pd.Timestamp(first.entry_day - timedelta(days=1))]
    notional = float(first.quantity * first.entry_price)
    assert 0.25 * sleeve - 2 * float(first.entry_price) <= notional <= 0.25 * sleeve * 1.01


def test_study_runs_every_stop_and_scope_on_both_halves():
    n = WARM + 200
    history = _core([100 + 0.1 * i for i in range(n)], n)
    history["AAPL"] = _series([10 + 0.01 * i for i in range(n)], "AAPL")
    report = cs.study(history)
    assert len(report["runs"]) == len(cs.CORE_STOPS) * len(cs.KILL_SCOPES)
    tune = [r["tune"]["money_weighted"] for r in report["runs"]]
    assert tune == sorted(tune, reverse=True)
    assert report["tune"][0] == _day(200).isoformat()
    assert report["validate"][1] == _day(n - 1).isoformat()
    assert set(report["benchmark"]) == {"tune", "validate"}
    assert "every deposit into VOO" in cs.format_table(report)


def test_the_sweep_parks_idle_swing_cash_in_voo_and_funds_buys_from_it():
    """D-043: with the sweep, swing cash above the buffer sits in VOO, so more of
    the account is invested; a swing buy still happens (funded by selling VOO)."""
    n = WARM + 160
    history = _core([100 + 0.05 * i for i in range(n)], n)
    history["AAPL"] = _series([10 + 0.05 * i for i in range(n)], "AAPL")
    args = (history, _day(WARM), _day(n - 1), CoreStop(), "swing only", backtest_limits())
    plain = cs.simulate_book(*args, sleeve_breakers=True)
    swept = cs.simulate_book(*args, sleeve_breakers=True, sweep=True)
    assert plain.sweep_trades == 0 and swept.sweep_trades > 0
    assert swept.invested.mean() > plain.invested.mean()
    buys = {t.symbol for t in swept.swing_trades}
    assert buys <= {"AAPL"}
    # Swing equity includes the parked VOO, so it is never below the parked value.
    assert (swept.swing_equity > 0).all()


def test_sleeve_breakers_measure_the_sleeve_not_the_account():
    """A core crash that leaves the swing sleeve flat must not trip the swing
    kill switch under D-042's rules (it does under D-039's account-wide ones)."""
    n = WARM + 120
    crash = [100.0] * (WARM + 40) + [60.0] * 80  # the core falls 40%
    history = _core(crash, n, other=100.0)
    args = (history, _day(WARM), _day(n - 1), CoreStop(), "swing only", backtest_limits())
    assert cs.simulate_book(*args).halted_on is not None
    assert cs.simulate_book(*args, sleeve_breakers=True).halted_on is None


def test_option_b_keeps_parked_voo_out_of_the_swing_kill_switch():
    """D-043 (b): a VOO crash drags the parked swing cash down. Measured with it,
    the sleeve's kill switch trips; with parked VOO left out, it does not."""
    n = WARM + 120
    crash = [100.0] * (WARM + 40) + [60.0] * 80  # VOO falls 40%, no swing symbols
    history = _core(crash, n)
    args = (history, _day(WARM), _day(n - 1), CoreStop(), "swing only", backtest_limits())
    swept = cs.simulate_book(*args, sleeve_breakers=True, sweep=True)
    assert swept.sweep_trades > 0 and swept.halted_on is not None
    kept_out = cs.simulate_book(*args, sleeve_breakers=True, sweep=True, parked_out=True)
    assert kept_out.halted_on is None


# --- live core rules vs the backtest -----------------------------------------------------


def _over_cap_state():
    """$1,000 then $500 deposited; crypto bought for $100 and now worth $200 (over its cap).

    Returns:
        (the ledger, the live portfolio, prices, values the backtest sees).
    """
    from trading_agent.risk.models import PortfolioState, Position
    from trading_agent.storage.models import CashFlow

    start = datetime(2026, 10, 1, tzinfo=UTC)
    rows = [("baseline", "stock", "-100000", "0")] + [
        ("deposit", venue, amount, core_part)
        for venue, amount, core_part in (
            ("crypto", "100", "100"),
            ("stock", "900", "600"),
            ("crypto", "50", "50"),
            ("stock", "450", "300"),
        )
    ]
    flows = [
        CashFlow(
            ts=start + timedelta(minutes=i),
            trading_mode="paper",
            asset_class=venue,
            kind=kind,
            amount=D(amount),
            core_amount=D(core_part),
            equity_before=D(0),
            scheduled_for=None if kind == "baseline" else start.date(),
        )
        for i, (kind, venue, amount, core_part) in enumerate(rows)
    ]
    btc = Position(
        symbol="BTC-USD", asset_class="crypto", quantity=D("0.002"), avg_entry_price=D(50000)
    )
    portfolio = PortfolioState(
        equity=D(1600),
        cash=D(1400),
        peak_equity=D(1600),
        day_start_equity=D(1600),
        week_start_equity=D(1600),
        consecutive_losses=0,
        positions={"BTC-USD": btc},
        venue_cash={"stock": D(1350), "crypto": D(50)},
    )
    prices = {"VOO": D(100), "QQQM": D(100), "BTC-USD": D(100000), "ETH-USD": D(3000)}
    values = {"VOO": D(0), "QQQM": D(0), "BTC-USD": D(200), "ETH-USD": D(0)}
    return flows, portfolio, prices, values


def test_the_backtest_buys_what_the_live_plan_buys_when_crypto_is_over_its_cap():
    """The crypto venue's $50 cannot reach stocks: stocks get 60% of $1,500, less
    the 1% headroom, in the backtest as live."""
    from trading_agent.portfolio import core

    flows, portfolio, prices, values = _over_cap_state()
    _, audit = core.plan(
        portfolio, prices, {}, {}, flows, [], datetime(2026, 10, 2, 15, tzinfo=UTC)
    )
    live = {s: D(v) for s, v in audit["buys"].items()}
    venues = {"stock": D(1350), "crypto": D(50)}
    backtest = core_buys(D(900), venues, values, set(cs.CORE_TARGETS), D(1600))
    assert sum(live.values()) == D(891)
    assert backtest == live
