"""The core sleeve's rules (D-040/D-041): allocation, stop, re-entry, cash."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pytest

from trading_agent.data.models import Candle
from trading_agent.portfolio import core
from trading_agent.risk.models import PortfolioState, Position
from trading_agent.storage.models import CashFlow, Order, Trade

START = datetime(2026, 10, 1, tzinfo=UTC)  # the ledger's baseline
NOW = datetime(2026, 11, 10, 15, tzinfo=UTC)
TODAY = NOW.replace(hour=0)
PRICES = {"VOO": D(100), "QQQM": D(50), "BTC-USD": D(60000), "ETH-USD": D(3000)}


def _flows(*deposits: tuple[str, str, str]) -> list[CashFlow]:
    """The baseline, then a deposit row per (venue, amount, core amount), a minute apart."""
    rows = [_flow("baseline", "stock", "-100000", START)]
    for i, (venue, amount, core_part) in enumerate(deposits):
        rows.append(_flow("deposit", venue, amount, START + timedelta(minutes=i), core_part))
    return rows


def _flow(kind: str, venue: str, amount: str, at: datetime, core_part: str = "0") -> CashFlow:
    """One unsaved CashFlow row."""
    return CashFlow(
        ts=at,
        trading_mode="paper",
        asset_class=venue,
        kind=kind,
        amount=D(amount),
        core_amount=D(core_part),
        equity_before=D(0),
    )


FIRST = _flows(("crypto", "100", "100"), ("stock", "900", "600"))  # the first $1,000


def _fill(symbol: str, side: str, qty: str, price: str, at: datetime, fees: str = "0") -> Trade:
    """One unsaved fill."""
    return Trade(
        filled_at=at,
        symbol=symbol,
        asset_class="crypto" if "-" in symbol else "stock",
        side=side,
        quantity=D(qty),
        fill_price=D(price),
        fees=D(fees),
        broker="test",
        trading_mode="paper",
    )


def _open_buy(symbol: str, qty: str, side: str = "buy") -> Order:
    """An open core market order (unsaved)."""
    return Order(
        created_at=NOW,
        symbol=symbol,
        asset_class="stock",
        side=side,
        quantity=D(qty),
        order_type="market",
        broker="test",
    )


def _bars(symbol: str, closes: list[float], end: datetime = TODAY) -> list[Candle]:
    """Daily candles, oldest first, the last one dated the day before `end`."""
    first = end - timedelta(days=len(closes))
    return [
        Candle(
            symbol=symbol,
            asset_class="stock",
            ts=first + timedelta(days=i),
            open=D(str(c)),
            high=D(str(c)),
            low=D(str(c)),
            close=D(str(c)),
            volume=D(1),
            source="test",
        )
        for i, c in enumerate(closes)
    ]


def _portfolio(stock_cash: str, crypto_cash: str, equity: str, positions=()) -> PortfolioState:
    """A portfolio on ledger terms; positions are (symbol, qty, avg) tuples."""
    return PortfolioState(
        equity=D(equity),
        cash=D(stock_cash) + D(crypto_cash),
        peak_equity=D(equity),
        day_start_equity=D(equity),
        week_start_equity=D(equity),
        consecutive_losses=0,
        positions={
            s: Position(
                symbol=s,
                asset_class="crypto" if "-" in s else "stock",
                quantity=D(q),
                avg_entry_price=D(a),
            )
            for s, q, a in positions
        },
        venue_cash={"stock": D(stock_cash), "crypto": D(crypto_cash)},
    )


def _plan(portfolio, prices=None, candles=None, fills=None, flows=FIRST, open_buys=()):
    """Run core.plan with defaults; return (orders by (symbol, side), audit)."""
    orders, audit = core.plan(
        portfolio,
        PRICES if prices is None else prices,
        candles or {},
        fills or {},
        flows,
        list(open_buys),
        NOW,
    )
    return {(o.symbol, o.side): o for o in orders}, audit


# --- allocate ---------------------------------------------------------------------


def test_allocate_fills_shortfalls_in_proportion():
    w = {"A": D(3), "B": D(1)}
    assert core.allocate(D(100), w, {}, {"A", "B"}) == {"A": D(75), "B": D(25)}


def test_an_overweight_holding_gets_nothing_until_the_others_catch_up():
    w = {"A": D(1), "B": D(1)}
    # Targets 75 each; A already holds 100, so all 50 go to B.
    assert core.allocate(D(50), w, {"A": D(100)}, {"A", "B"}) == {"B": D(50)}


def test_shortfalls_are_measured_on_cash_plus_holdings():
    w = {"A": D(1), "B": D(1)}
    # Targets 55 each: shortfalls 45 and 55 use all 100.
    assert core.allocate(D(100), w, {"A": D(10)}, {"A", "B"}) == {"A": D(45), "B": D(55)}
    # Pending buys count as held value.
    assert core.allocate(D(10), w, {"A": D(40), "B": D(50)}, {"A", "B"}) == {"A": D(10)}
    # With an overweight holding the shortfalls exceed the cash: scaled down.
    got = core.allocate(D(30), w, {"A": D(0), "B": D(90)}, {"A", "B"})
    assert got == {"A": D(30)}  # targets 60: A short 60, B over; all 30 to A


def test_an_ineligible_holding_keeps_its_share_in_cash():
    w = {"A": D(3), "B": D(1)}
    assert core.allocate(D(100), w, {}, {"B"}) == {"B": D(25)}


def test_amounts_below_the_minimum_order_wait():
    w = {"A": D(1), "B": D(1)}
    assert core.allocate(D(15), w, {}, {"A", "B"}) == {}  # 7.50 each
    assert core.allocate(D(0), w, {}, {"A", "B"}) == {}


# --- crossed_back -----------------------------------------------------------------

LINE_BARS = [100.0] * 200  # the 200-day line sits at exactly 100


def _sold_at(bars: list[Candle], index: int) -> datetime:
    """Noon of bar `index`'s day: a sale during that bar."""
    return bars[index].ts + timedelta(hours=12)


def test_a_cross_needs_a_close_at_or_below_the_line_then_one_above():
    bars = _bars("VOO", [*LINE_BARS, 100.0, 101.0])  # at the line, then above
    assert core.crossed_back(bars, _sold_at(bars, 199))


def test_a_sale_above_the_line_does_not_buy_back_while_it_stays_above():
    bars = _bars("VOO", [*LINE_BARS, 105.0, 106.0, 107.0])
    assert not core.crossed_back(bars, _sold_at(bars, 200))


def test_a_dip_before_the_sale_does_not_count():
    bars = _bars("VOO", [*LINE_BARS[:-1], 90.0, 120.0, 125.0])
    assert not core.crossed_back(bars, _sold_at(bars, 200))


def test_below_the_line_and_still_there_is_no_cross():
    bars = _bars("VOO", [*LINE_BARS, 95.0, 96.0])
    assert not core.crossed_back(bars, _sold_at(bars, 199))


def test_the_sale_days_own_close_counts_toward_the_cross():
    bars = _bars("VOO", [*LINE_BARS, 100.0, 101.0])  # the sale's day closes at the line
    assert core.crossed_back(bars, _sold_at(bars, 200))


def test_no_cross_while_the_line_cannot_be_computed():
    bars = _bars("BTC-USD", [100.0] * 50 + [90.0, 110.0])
    assert not core.crossed_back(bars, _sold_at(bars, 10))


def test_average_closes_is_the_trailing_mean():
    bars = _bars("VOO", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert core.average_closes(bars, days=3) == [None, None, D(2), D(3), D(4)]


def test_the_trail_sells_25_percent_below_the_high_floored_at_cost():
    assert core.trail_breached(D(90), D(100), [D(120)])
    assert not core.trail_breached(D("90.01"), D(100), [D(120)])
    assert core.trail_breached(D(75), D(100), [])  # no history: the high is the cost
    assert not core.trail_breached(D(80), D(100), [D(90)])


def test_budgets_keep_each_venues_cash_apart():
    cash = core.budgets(D(600), {"stock": D(900), "crypto": D(100)}, D(0), D(1000))
    assert cash == {"stock": D(600) * D("0.99"), "crypto": D(100) * D("0.99")}
    over = core.budgets(D(600), {"stock": D(900), "crypto": D(100)}, D(150), D(1000))
    assert over["crypto"] == 0 and over["stock"] == D(600) * D("0.99")


# --- stock_core_cash ----------------------------------------------------------------


def test_stock_core_cash_is_the_core_share_of_deposits_less_what_it_spent():
    after = START + timedelta(days=1)
    fills = [
        _fill("VOO", "buy", "1", "999", START - timedelta(days=1)),  # before the ledger: ignored
        _fill("VOO", "buy", "2", "100", after, fees="1"),  # -201
        _fill("QQQM", "sell", "1", "50", after, fees="0.5"),  # +49.5
    ]
    # 60% of 1,000 deposited (both venues' rows) = 600; 100 is reserved by an open buy.
    got = core.stock_core_cash(FIRST, fills, D(100))
    assert got == D(600) - D(201) + D("49.5") - D(100)


def test_the_core_stock_targets_add_up_to_the_share_each_deposit_records():
    from trading_agent.portfolio import ledger

    stock_targets = sum((core.CORE_TARGETS[s] for s in core.CORE_STOCKS), D(0))
    assert stock_targets == ledger.CORE_STOCK_SHARE


def test_a_changed_core_share_never_moves_money_already_deposited(monkeypatch):
    """The split is recorded when a deposit is booked: a later change to the share
    applies to later deposits only."""
    from trading_agent.portfolio import ledger

    rows = [
        _flow("baseline", "stock", "-100000", START),
        *ledger.deposit_rows(D(1000), {"stock", "crypto"}, D(0), START, START.date(), "paper"),
    ]
    before = core.stock_core_cash(rows, [], D(0))
    monkeypatch.setattr(ledger, "CORE_STOCK_SHARE", D("0.50"))
    assert core.stock_core_cash(rows, [], D(0)) == before == D(600)


# --- plan: deposit buys -----------------------------------------------------------


def test_the_first_deposit_buys_every_holding_toward_its_weight():
    orders, audit = _plan(_portfolio("900", "100", "1000"))
    stock = D(600) * D("0.99")  # the core stock share, venue_cash_usable_pct
    crypto = D(100) * D("0.99")
    assert {k: D(v) for k, v in audit["buys"].items()} == {
        "BTC-USD": crypto * D("0.6"),
        "ETH-USD": crypto * D("0.4"),
        "VOO": stock * D("0.625"),
        "QQQM": stock * D("0.375"),
    }
    voo = orders[("VOO", "buy")]
    assert voo.quantity == stock * D("0.625") / D(100)
    assert (voo.stop_loss_price, voo.source_proposal, voo.asset_class) == (None, None, "stock")
    assert orders[("BTC-USD", "buy")].asset_class == "crypto"


def test_no_stock_buys_outside_the_session_but_crypto_still_buys():
    prices = {"BTC-USD": D(60000), "ETH-USD": D(3000)}
    orders, _ = _plan(_portfolio("900", "100", "1000"), prices=prices)
    assert set(orders) == {("BTC-USD", "buy"), ("ETH-USD", "buy")}


def test_crypto_at_the_cap_buys_nothing_and_sells_nothing():
    held = [("BTC-USD", "0.002", "60000")]  # 120 of a 1,000 account: over the 10% cap
    orders, audit = _plan(_portfolio("900", "100", "1000", held))
    assert not [k for k in orders if k[0] in core.CORE_CRYPTO]
    assert audit["cash"]["crypto"] == "0"


def test_crypto_buys_stop_at_the_cap():
    held = [("BTC-USD", "0.001", "60000")]  # 60 of 1,000: 40 of room left
    _, audit = _plan(_portfolio("900", "100", "1000", held))
    assert D(audit["cash"]["crypto"]) == D(40)


def test_each_venue_pays_only_from_its_own_cash():
    _, audit = _plan(_portfolio("200", "100", "1000"))
    assert D(audit["cash"]["stock"]) == D(200) * D("0.99")  # less than the 600 core share
    _, audit = _plan(_portfolio("900", "5", "1000"))
    assert D(audit["cash"]["crypto"]) == D(5) * D("0.99")


def test_open_buys_are_spoken_for():
    _, audit = _plan(_portfolio("900", "100", "1000"), open_buys=[_open_buy("VOO", "3")])
    # 600 - 300 reserved = 297 usable. VOO's 300 pending counts toward its weight:
    # targets on 597 are 373.1 / 223.9, so VOO needs 73.1 and QQQM 223.9.
    assert D(audit["cash"]["stock"]) == D(300) * D("0.99")
    voo, qqqm = D(audit["buys"]["VOO"]), D(audit["buys"]["QQQM"])
    assert voo < D(80) < D(200) < qqqm


def test_an_open_sell_is_a_sale_already_made():
    """The executor books a sell's fill only on a later cycle: until then the
    book shows the buy and the venue no position. Buying again would undo
    the stop-out without waiting for the 200-day cross."""
    candles = {"VOO": _bars("VOO", [*LINE_BARS, 105.0])}
    fills = {"VOO": [_fill("VOO", "buy", "3", "130", START + timedelta(days=1))]}
    orders, audit = _plan(
        _portfolio("900", "100", "1000"),
        candles=candles,
        fills=fills,
        open_buys=[_open_buy("VOO", "3", side="sell")],
    )
    assert audit["waiting"] == ["VOO"] and ("VOO", "buy") not in orders
    # ...and a holding still showing at the venue is not sold twice.
    held = _portfolio("0", "0", "1000", [("VOO", "3", "130")])
    orders, _ = _plan(held, candles=candles, fills=fills, open_buys=[_open_buy("VOO", "3", "sell")])
    assert ("VOO", "sell") not in orders


# --- plan: the 25% trail -----------------------------------------------------------

OPENED = TODAY - timedelta(days=10)


def _held_voo(closes: list[float], avg: str = "100") -> tuple[dict, dict, PortfolioState]:
    """VOO held since OPENED, with complete closes up to yesterday: (candles, fills, portfolio)."""
    candles = {"VOO": _bars("VOO", closes)}
    fills = {"VOO": [_fill("VOO", "buy", "3", avg, OPENED)]}
    portfolio = _portfolio("0", "0", "1000", [("VOO", "3", avg)])
    return candles, fills, portfolio


@pytest.mark.parametrize(("last", "sells"), [(90.0, True), (90.01, False)])
def test_the_trail_sells_at_exactly_25_percent_below_the_high(last, sells):
    candles, fills, portfolio = _held_voo([100.0] * 5 + [120.0, 110.0, 100.0, 95.0, last])
    orders, audit = _plan(portfolio, candles=candles, fills=fills)
    assert (("VOO", "sell") in orders) is sells
    if sells:
        assert orders[("VOO", "sell")].quantity == D(3)
        assert audit["sells"] == [{"symbol": "VOO", "close": "90.0", "high": "120.0"}]


def test_the_trail_high_is_floored_at_the_average_cost():
    candles, fills, portfolio = _held_voo([150.0] * 9 + [150.0], avg="200")
    orders, _ = _plan(portfolio, candles=candles, fills=fills)
    assert ("VOO", "sell") in orders  # 150 = 75% of the 200 cost


def test_the_opening_days_close_counts_toward_the_high():
    closes = [100.0] * 20 + [140.0] + [100.0] * 9  # the 140 is OPENED's own close
    candles = {"VOO": _bars("VOO", closes)}
    fills = {"VOO": [_fill("VOO", "buy", "3", "100", OPENED + timedelta(hours=15))]}
    held = _portfolio("0", "0", "1000", [("VOO", "3", "100")])
    _, audit = _plan(held, candles=candles, fills=fills)
    assert audit["sells"][0]["high"] == "140.0"  # 100 <= 75% of 140


def test_highs_before_the_position_opened_do_not_count():
    closes = [200.0] * 20 + [100.0] * 10  # the 200s are before OPENED
    candles = {"VOO": _bars("VOO", closes)}
    fills = {"VOO": [_fill("VOO", "buy", "3", "100", OPENED)]}
    orders, _ = _plan(
        _portfolio("0", "0", "1000", [("VOO", "3", "100")]), candles=candles, fills=fills
    )
    assert ("VOO", "sell") not in orders


def test_todays_forming_bar_never_triggers_the_trail():
    candles, fills, portfolio = _held_voo([100.0] * 10)
    candles["VOO"] += _bars("VOO", [50.0], end=TODAY + timedelta(days=1))  # dated today
    orders, _ = _plan(portfolio, candles=candles, fills=fills)
    assert ("VOO", "sell") not in orders


def test_no_price_this_cycle_means_no_sell():
    candles, fills, portfolio = _held_voo([100.0] * 5 + [120.0, 80.0])
    prices = {k: v for k, v in PRICES.items() if k != "VOO"}
    orders, _ = _plan(portfolio, prices=prices, candles=candles, fills=fills)
    assert ("VOO", "sell") not in orders


def test_a_holding_with_no_fill_history_trails_from_its_cost():
    candles = {"VOO": _bars("VOO", [120.0] * 5 + [75.0])}
    orders, _ = _plan(_portfolio("0", "0", "1000", [("VOO", "3", "100")]), candles=candles)
    assert ("VOO", "sell") in orders  # 75 = 75% of the 100 cost; the 120s are not trusted


def test_a_symbol_sold_this_cycle_is_not_bought_this_cycle():
    candles, fills, portfolio = _held_voo([120.0, 80.0])
    portfolio.venue_cash["stock"] = D(900)
    orders, _ = _plan(portfolio, candles=candles, fills=fills)
    assert ("VOO", "sell") in orders and ("VOO", "buy") not in orders


# --- plan: re-entry -------------------------------------------------------------------


def _sold_voo(after_sale: list[float], sale_day: int = 200) -> tuple[dict, dict]:
    """VOO bought, then sold by the trail on bar `sale_day`, with 200 flat bars for the line."""
    bars = _bars("VOO", [*LINE_BARS, 105.0, *after_sale])
    sold = _sold_at(bars, sale_day)
    fills = {
        "VOO": [
            _fill("VOO", "buy", "3", "130", sold - timedelta(days=30)),
            _fill("VOO", "sell", "3", "97", sold),
        ]
    }
    return {"VOO": bars}, fills


def test_a_sold_holding_waits_for_the_cross_and_its_share_waits_in_cash():
    candles, fills = _sold_voo([106.0, 107.0])  # never back to the line
    flows = _flows(("crypto", "100", "100"), ("stock", "900", "600"))
    flows[0].ts = START - timedelta(days=400)  # the ledger predates the sale
    orders, audit = _plan(
        _portfolio("900", "100", "1000"), candles=candles, fills=fills, flows=flows
    )
    assert audit["waiting"] == ["VOO"]
    assert ("VOO", "buy") not in orders
    # QQQM gets only its own shortfall (22.5% of the account), not VOO's share.
    assert D(audit["buys"]["QQQM"]) < D(600) * D("0.99") * D("0.375") + D(1)


def test_a_sold_holding_buys_again_after_the_cross():
    candles, fills = _sold_voo([100.0, 101.0])
    flows = _flows(("crypto", "100", "100"), ("stock", "900", "600"))
    flows[0].ts = START - timedelta(days=400)
    orders, audit = _plan(
        _portfolio("900", "100", "1000"), candles=candles, fills=fills, flows=flows
    )
    assert audit["waiting"] == [] and ("VOO", "buy") in orders


def test_the_latest_of_several_sales_decides():
    candles, fills = _sold_voo([100.0, 101.0, 106.0])  # crossed after the first sale...
    sold = candles["VOO"][-1].ts + timedelta(hours=12)  # ...then sold again yesterday
    fills["VOO"] += [_fill("VOO", "buy", "3", "110", sold - timedelta(hours=6))]
    fills["VOO"] += [_fill("VOO", "sell", "3", "106", sold)]
    flows = _flows(("crypto", "100", "100"), ("stock", "900", "600"))
    flows[0].ts = START - timedelta(days=400)
    _, audit = _plan(_portfolio("900", "100", "1000"), candles=candles, fills=fills, flows=flows)
    assert audit["waiting"] == ["VOO"]


def test_a_sale_before_the_ledger_started_never_blocks_buying():
    # Rising closes stay above their own 200-day line: no cross since any sale.
    bars = _bars("VOO", [float(v) for v in range(1, 211)])
    candles = {"VOO": bars}
    fills = {"VOO": [_fill("VOO", "sell", "3", "97", bars[0].ts - timedelta(days=1))]}
    orders, audit = _plan(_portfolio("900", "100", "1000"), candles=candles, fills=fills)
    assert audit["waiting"] == [] and ("VOO", "buy") in orders


def test_the_200_day_line_matches_a_rolling_mean():
    """The running-sum line against pandas' rolling mean, over varying prices."""
    import random

    import pandas as pd

    rng = random.Random(7)
    closes = [100.0]
    for _ in range(299):
        closes.append(round(closes[-1] * (1 + rng.uniform(-0.03, 0.03)), 2))
    bars = _bars("VOO", closes)
    line = pd.Series(closes).rolling(200).mean()
    for sale in (150, 205, 240, 280):
        below, expected = False, False
        for i in range(max(sale, 199), len(closes)):
            if closes[i] <= line[i] + 1e-9:
                below = True
            elif below:
                expected = True
                break
        assert core.crossed_back(bars, _sold_at(bars, sale)) is expected, sale


def test_holdings_are_valued_at_market_for_the_weights():
    """VOO bought at 50 now trades at 100: it counts as 400, not its 200 cost."""
    held = _portfolio("900", "100", "1400", [("VOO", "4", "50")])
    _, audit = _plan(held, candles={"VOO": _bars("VOO", [100.0] * 5)})
    # 594 usable: VOO's target is 62.5% of (594 + 400) = 621.25, so it is 221.25 short.
    assert D(audit["buys"]["VOO"]) == D("221.25")
