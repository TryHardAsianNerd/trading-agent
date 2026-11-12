"""Gate tests for the BATCH dimension.

The LLM controls how many proposals it emits, so every limit must hold across
the whole batch, not per order, and no single proposal may crash or steer the gate.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from fakes import BrokenRedis, FakeRedis
from trading_agent.data.models import Candle
from trading_agent.graph.state import TradeProposal
from trading_agent.risk.gate import evaluate
from trading_agent.risk.killswitch import KILL_SWITCH_KEY, KillSwitch
from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.models import (
    MarketEnvironment,
    PortfolioState,
    Position,
    RejectReason,
)

LIMITS = RiskLimits()
T0 = datetime(2026, 1, 1, tzinfo=UTC)
ENV_OK = MarketEnvironment()


def _candles(
    symbol: str, n: int = 20, asset_class: str = "stock", volume: Decimal | None = None
) -> list[Candle]:
    """Liquid-by-default candles: stock volume is shares, crypto volume is USD (D-022)."""
    if volume is None:
        volume = Decimal(100_000_000) if asset_class == "crypto" else Decimal(1_000_000)
    return [
        Candle(
            symbol=symbol,
            asset_class=asset_class,
            ts=T0 + timedelta(days=i),
            open=Decimal(100),
            high=Decimal(101),
            low=Decimal(99),
            close=Decimal(100),
            volume=volume,
            source="test",
        )
        for i in range(n)
    ]


def _portfolio(**overrides) -> PortfolioState:
    base = dict(
        equity=Decimal(100_000),
        cash=Decimal(100_000),
        peak_equity=Decimal(100_000),
        day_start_equity=Decimal(100_000),
        week_start_equity=Decimal(100_000),
        consecutive_losses=0,
        positions={},
    )
    base.update(overrides)
    return PortfolioState(**base)


def _buy(symbol: str) -> TradeProposal:
    return TradeProposal(
        symbol=symbol, asset_class="stock", action="buy", confidence=0.8, reasoning="r"
    )


def _sell(symbol: str, asset_class: str = "stock") -> TradeProposal:
    return TradeProposal(
        symbol=symbol, asset_class=asset_class, action="sell", confidence=0.8, reasoning="r"
    )


def _run(proposals, portfolio, ks=None, symbols=None, env=ENV_OK):
    symbols = symbols or sorted({p.symbol for p in proposals})
    return evaluate(
        proposals,
        portfolio,
        {s: _candles(s) for s in symbols},
        {s: Decimal(100) for s in symbols},
        env,
        ks or KillSwitch(FakeRedis()),
        LIMITS,
    )


# --- batch cash accounting -------------------------------------------------


def test_batch_cannot_breach_cash_reserve():
    """20 buys with only $10k deployable: aggregate notional must be <= $10k."""
    p = _portfolio(cash=Decimal(30_000))  # reserve 20k -> deployable 10k
    proposals = [_buy(f"S{i:02d}") for i in range(20)]
    d = _run(proposals, p)
    total_notional = sum(o.quantity * Decimal(100) for o in d.approved)
    assert total_notional <= Decimal(10_000)
    # And the rest were rejected for cash, not silently dropped.
    assert len(d.approved) + len(d.rejected) == 20
    assert any(r.reason is RejectReason.INSUFFICIENT_CASH_AFTER_RESERVE for r in d.rejected)


def test_batch_aggregate_risk_bounded():
    """Aggregate loss-at-stop across a full-cash batch stays bounded by the
    per-symbol caps: each approved order still risks <= 1% of equity."""
    p = _portfolio()
    d = _run([_buy(f"S{i:02d}") for i in range(20)], p)
    for o in d.approved:
        loss_at_stop = o.quantity * (Decimal(100) - o.stop_loss_price)
        assert loss_at_stop <= p.equity * LIMITS.risk_per_trade_pct


# --- duplicates --------------------------------------------------------------


def test_duplicate_sells_cannot_go_net_short():
    p = _portfolio(
        positions={
            "AAPL": Position(
                symbol="AAPL",
                asset_class="stock",
                quantity=Decimal(10),
                avg_entry_price=Decimal(90),
            )
        }
    )
    d = _run([_sell("AAPL"), _sell("AAPL"), _sell("AAPL")], p)
    assert len(d.approved) == 1
    assert d.approved[0].quantity == Decimal(10)
    assert all(r.reason is RejectReason.DUPLICATE_PROPOSAL for r in d.rejected)


def test_duplicate_buys_approved_once():
    d = _run([_buy("AAPL"), _buy("AAPL"), _buy("AAPL")], _portfolio())
    assert len(d.approved) == 1
    assert d.approved[0].quantity == Decimal(100)
    assert len(d.rejected) == 2


# --- existing positions count toward the cap --------------------------------


@pytest.mark.parametrize("held", ["100", "30"], ids=["at-cap", "headroom-left"])
def test_a_held_symbol_is_never_added_to(held):
    """Buys open positions only (D-038): the tested entry buys when flat, and a
    top-up of a whole-share position rounds to 0 at Alpaca and pages."""
    p = _portfolio(
        positions={
            "AAPL": Position(
                symbol="AAPL",
                asset_class="stock",
                quantity=Decimal(held),
                avg_entry_price=Decimal(100),
            )
        }
    )
    d = _run([_buy("AAPL")], p)
    assert d.approved == []
    assert d.rejected[0].reason is RejectReason.ALREADY_HELD


def test_zero_atr_rejects_proposal_and_batch_survives():
    flat = [
        Candle(
            symbol="FLAT",
            asset_class="stock",
            ts=T0 + timedelta(days=i),
            open=Decimal(100),
            high=Decimal(100),
            low=Decimal(100),
            close=Decimal(100),
            volume=Decimal(1_000_000),
            source="test",
        )
        for i in range(20)
    ]
    d = evaluate(
        [_buy("FLAT"), _buy("AAPL")],
        _portfolio(),
        {"FLAT": flat, "AAPL": _candles("AAPL")},
        {"FLAT": Decimal(100), "AAPL": Decimal(100)},
        ENV_OK,
        KillSwitch(FakeRedis()),
        LIMITS,
    )
    assert len(d.approved) == 1 and d.approved[0].symbol == "AAPL"
    assert d.rejected[0].reason is RejectReason.INSUFFICIENT_DATA


# --- asset_class is never taken from the LLM --------------------------------


def test_sell_asset_class_comes_from_position_not_proposal():
    p = _portfolio(
        positions={
            "AAPL": Position(
                symbol="AAPL",
                asset_class="stock",
                quantity=Decimal(10),
                avg_entry_price=Decimal(90),
            )
        }
    )
    d = _run([_sell("AAPL", asset_class="crypto")], p)  # LLM lies about the venue
    assert d.approved[0].asset_class == "stock"


def test_buy_asset_class_comes_from_market_data_not_proposal():
    proposal = TradeProposal(
        symbol="BTC-USD", asset_class="stock", action="buy", confidence=0.8, reasoning="r"
    )  # LLM claims stock; the candles say crypto
    d = evaluate(
        [proposal],
        _portfolio(),
        {"BTC-USD": _candles("BTC-USD", asset_class="crypto")},
        {"BTC-USD": Decimal(100)},
        ENV_OK,
        KillSwitch(FakeRedis()),
        LIMITS,
    )
    assert d.approved[0].asset_class == "crypto"


# --- Redis write failures never crash the gate ------------------------------


def test_drawdown_with_broken_redis_still_rejects_without_crashing():
    ks = KillSwitch(BrokenRedis())
    p = _portfolio(
        equity=Decimal(84_000),
        day_start_equity=Decimal(84_000),
        week_start_equity=Decimal(84_000),
    )
    d = _run([_buy("AAPL")], p, ks=ks)
    # activate() failed silently; the batch still rejects on the drawdown reason itself.
    assert d.rejected[0].reason is RejectReason.MAX_DRAWDOWN_EXCEEDED


# --- weekly breaker pauses for the rest of the week -------------------------


def test_weekly_breaker_cooldown_extends_to_week_end():
    fake = FakeRedis()
    ks = KillSwitch(fake)
    p = _portfolio(
        equity=Decimal(95_000), day_start_equity=Decimal(95_000)
    )  # weekly -5%, daily flat
    _run([_buy("AAPL")], p, ks=ks)
    from trading_agent.risk.killswitch import COOLDOWN_KEY

    assert fake.ttls[COOLDOWN_KEY] >= 3600
    assert fake.ttls[COOLDOWN_KEY] <= 7 * 24 * 3600


# --- drawdown during an active cooldown still arms the kill switch ----------


def test_drawdown_arms_kill_switch_even_during_cooldown():
    fake = FakeRedis()
    ks = KillSwitch(fake)
    ks.start_cooldown("daily_loss_breaker", LIMITS)
    p = _portfolio(
        equity=Decimal(84_000),
        day_start_equity=Decimal(84_000),
        week_start_equity=Decimal(84_000),
    )
    d = _run([_buy("AAPL")], p, ks=ks)
    assert d.rejected[0].reason is RejectReason.MAX_DRAWDOWN_EXCEEDED
    assert ks.is_halted()[1] == "kill_switch:max_drawdown_exceeded"


# --- duplicate candle timestamps don't deflate ATR --------------------------


def test_duplicate_timestamp_candles_do_not_shrink_stops():
    from trading_agent.risk.stops import wilder_atr

    clean = _candles("AAPL")
    doubled = clean + clean  # every row duplicated
    assert wilder_atr(doubled) == wilder_atr(clean)


# --- an ILLIQUID rejection must not consume batch headroom (D-022) ----------


def test_illiquid_rejection_consumes_no_cash():
    """A thin symbol proposed first must leave the full deployable cash for the
    liquid names behind it: the filter runs before any sizing."""
    p = _portfolio(cash=Decimal(30_000))  # reserve 20k -> deployable 10k
    candles = {"THIN": _candles("THIN", volume=Decimal(1)), "AAPL": _candles("AAPL")}
    prices = {"THIN": Decimal(100), "AAPL": Decimal(100)}
    with_thin = evaluate(
        [_buy("THIN"), _buy("AAPL")],
        p,
        candles,
        prices,
        ENV_OK,
        KillSwitch(FakeRedis()),
        LIMITS,
    )
    without_thin = _run([_buy("AAPL")], p)
    assert [r.reason for r in with_thin.rejected] == [RejectReason.ILLIQUID]
    assert with_thin.approved[0].quantity == without_thin.approved[0].quantity


def test_illiquid_rejection_leaves_cap_headroom_intact_in_the_same_batch():
    """A liquid name must still size to its own cap when an illiquid proposal
    precedes it in the same batch."""
    p = _portfolio()
    solo = _run([_buy("AAPL")], p).approved[0].quantity
    d = evaluate(
        [_buy("THIN"), _buy("AAPL")],
        p,
        {"THIN": _candles("THIN", volume=Decimal(1)), "AAPL": _candles("AAPL")},
        {"THIN": Decimal(100), "AAPL": Decimal(100)},
        ENV_OK,
        KillSwitch(FakeRedis()),
        LIMITS,
    )
    assert [r.reason for r in d.rejected] == [RejectReason.ILLIQUID]
    assert d.audit["illiquid"] == {
        "THIN": "20d avg dollar volume $100.00 vs stock minimum $10,000,000.00"
    }
    assert [(o.symbol, o.quantity) for o in d.approved] == [("AAPL", solo)]


def test_weekly_breaker_pause_is_measured_from_the_given_clock():
    """The backtest runs the gate on a simulated clock: the rest-of-week pause
    must be measured from THAT time, not the wall clock."""
    fake = FakeRedis()
    ks = KillSwitch(fake)
    p = _portfolio(equity=Decimal(95_000), day_start_equity=Decimal(95_000))
    friday_evening = datetime(2023, 3, 10, 20, tzinfo=UTC)
    evaluate(
        [_buy("AAPL")],
        p,
        {"AAPL": _candles("AAPL")},
        {"AAPL": Decimal(100)},
        ENV_OK,
        ks,
        now=friday_evening,
    )
    from trading_agent.risk.killswitch import COOLDOWN_KEY

    assert fake.ttls[COOLDOWN_KEY] == 52 * 3600  # to Monday 2023-03-13 00:00 UTC


# --- per-venue cash: a buy can only spend its own venue's cash (D-032) ---------


def _venue_run(proposals, venue_cash):
    """Evaluate buys for stock AAPL and crypto BTC-USD / ETH-USD at price 100.

    Pooled equity and cash are 100k with a 20% reserve; returns the GateDecision.
    """
    candles = {
        "AAPL": _candles("AAPL"),
        "BTC-USD": _candles("BTC-USD", asset_class="crypto"),
        "ETH-USD": _candles("ETH-USD", asset_class="crypto"),
    }
    return evaluate(
        proposals,
        _portfolio(venue_cash=venue_cash),
        candles,
        dict.fromkeys(candles, Decimal(100)),
        ENV_OK,
        KillSwitch(FakeRedis()),
        LIMITS,
    )


def test_a_crypto_buy_is_capped_by_the_crypto_venues_own_cash():
    """Pooled cash would allow the 10% cap (10k); the crypto venue holds 5k,
    of which 99% is usable after fill costs -> 49.5 units, not 100."""
    d = _venue_run([_buy("BTC-USD")], {"stock": Decimal(95_000), "crypto": Decimal(5_000)})
    (order,) = d.approved
    assert order.quantity == Decimal("49.5")


def test_the_venue_remainder_carries_across_the_batch():
    d = _venue_run(
        [_buy("BTC-USD"), _buy("ETH-USD"), _buy("AAPL")],
        {"stock": Decimal(95_000), "crypto": Decimal(5_000)},
    )
    assert [(o.symbol, o.quantity) for o in d.approved] == [
        ("BTC-USD", Decimal("49.5")),
        ("AAPL", Decimal(100)),  # the stock venue is untouched by crypto spending
    ]
    assert [(r.proposal.symbol, r.reason) for r in d.rejected] == [
        ("ETH-USD", RejectReason.INSUFFICIENT_CASH_AFTER_RESERVE)
    ]


def test_no_venue_view_means_pooled_cash_only():
    (order,) = _venue_run([_buy("BTC-USD")], {}).approved
    assert order.quantity == Decimal(100)


def test_stock_buys_are_fractional_and_keep_their_stop():
    # 10% cap of 1,500 at 100 = 1.5 shares; 10% of 500 = half a share.
    assert [o.quantity for o in _run([_buy("AAPL")], _small(1500)).approved] == [Decimal("1.5")]
    (order,) = _run([_buy("AAPL")], _small(500)).approved
    assert order.quantity == Decimal("0.5") and order.stop_loss_price is not None


def test_a_buy_under_the_minimum_order_is_refused_without_paging():
    """10% of a $50 account is $5: under the $10 minimum, a routine sizing result."""
    d = _run([_buy("AAPL")], _small(50))
    assert d.approved == []
    (r,) = d.rejected
    assert r.reason is RejectReason.POSITION_SIZE_ZERO
    assert r.detail == "$5.00 is under the $10 minimum order"


def test_crypto_buys_stay_fractional():
    buy = TradeProposal(
        symbol="BTC-USD", asset_class="crypto", action="buy", confidence=0.8, reasoning="r"
    )
    d = evaluate(
        [buy],
        _small(500),
        {"BTC-USD": _candles("BTC-USD", asset_class="crypto")},
        {"BTC-USD": Decimal(100)},
        ENV_OK,
        KillSwitch(FakeRedis()),
        LIMITS,
    )
    assert [o.quantity for o in d.approved] == [Decimal("0.5")]


def test_the_drawdown_kill_switch_arms_once_and_never_overwrites_a_manual_one():
    """Cycles keep running under the automatic switch (D-042): re-arming every
    cycle would page every 15 minutes, and overwriting a human's switch would
    turn "stop everything" into "stop swing entries"."""
    deep = _portfolio(equity=Decimal(80_000), peak_equity=Decimal(100_000))

    class CountingRedis(FakeRedis):
        writes = 0

        def set(self, key, value, ex=None):
            CountingRedis.writes += key == KILL_SWITCH_KEY
            super().set(key, value, ex)

    ks = KillSwitch(CountingRedis())
    _run([_buy("AAPL")], deep, ks)
    d = _run([_buy("AAPL")], deep, ks)
    assert CountingRedis.writes == 1
    assert d.rejected[0].reason is RejectReason.MAX_DRAWDOWN_EXCEEDED
    manual = FakeRedis()
    manual.set(KILL_SWITCH_KEY, "manual_discord")
    _run([_buy("AAPL")], deep, KillSwitch(manual))
    assert manual.get(KILL_SWITCH_KEY) == "manual_discord"


def _small(amount: int) -> PortfolioState:
    """A flat account of `amount` equity and cash with no loss against any anchor."""
    v = Decimal(amount)
    return _portfolio(equity=v, cash=v, peak_equity=v, day_start_equity=v, week_start_equity=v)


def test_the_minimum_order_is_exactly_10_dollars_and_for_stocks_only():
    """10% of $100 is exactly $10: allowed. Crypto keeps its venue's own minimum."""
    (order,) = _run([_buy("AAPL")], _small(100)).approved
    assert order.quantity * Decimal(100) == Decimal(10)
    crypto = TradeProposal(
        symbol="BTC-USD", asset_class="crypto", action="buy", confidence=0.8, reasoning="r"
    )
    d = evaluate(
        [crypto],
        _small(50),
        {"BTC-USD": _candles("BTC-USD", asset_class="crypto")},
        {"BTC-USD": Decimal(100)},
        ENV_OK,
        KillSwitch(FakeRedis()),
        LIMITS,
    )
    assert [o.quantity * Decimal(100) for o in d.approved] == [Decimal(5)]
