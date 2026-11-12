"""Risk gate integration tests: all five layers through the single entry point."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from fakes import FakeRedis
from trading_agent.data.models import Candle
from trading_agent.graph.state import TradeProposal
from trading_agent.risk.gate import evaluate
from trading_agent.risk.killswitch import KillSwitch
from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.models import MarketEnvironment, PortfolioState, Position, RejectReason

LIMITS = RiskLimits()
T0 = datetime(2026, 1, 1, tzinfo=UTC)
ENV_OK = MarketEnvironment()


LIQUID_SHARES = Decimal(1_000_000)  # x close 100 = $100M/day, well over the stock minimum


def _candles(n: int = 20, volume: Decimal = LIQUID_SHARES, symbol: str = "AAPL") -> list[Candle]:
    """Constant true range of 2 → ATR exactly 2; liquid by default."""
    return [
        Candle(
            symbol=symbol,
            asset_class="stock",
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


def _buy(symbol: str = "AAPL", confidence: float = 0.8) -> TradeProposal:
    return TradeProposal(
        symbol=symbol, asset_class="stock", action="buy", confidence=confidence, reasoning="r"
    )


def _ks() -> KillSwitch:
    return KillSwitch(FakeRedis())


def _evaluate(proposals, portfolio, ks=None, env=ENV_OK, prices=None, candles=None):
    return evaluate(
        proposals,
        portfolio,
        candles if candles is not None else {"AAPL": _candles()},
        prices if prices is not None else {"AAPL": Decimal(100)},
        env,
        ks or _ks(),
        LIMITS,
    )


def test_buy_happy_path_exact_numbers():
    d = _evaluate([_buy()], _portfolio())
    assert d.rejected == []
    (order,) = d.approved
    # ATR 2 → stop = 100 - 3x2 = 94; risk 1000/6 = 166.7 → 10% cap → 100 units (D-032).
    assert order.side == "buy"
    assert order.stop_loss_price == Decimal(94)
    assert order.quantity == Decimal(100)
    assert order.source_proposal.confidence == 0.8


def test_hold_produces_nothing_but_is_audited():
    hold = TradeProposal(
        symbol="MSFT", asset_class="stock", action="hold", confidence=0.5, reasoning="r"
    )
    d = _evaluate([hold], _portfolio())
    assert d.approved == [] and d.rejected == []
    assert d.audit["holds"] == ["MSFT"]


def test_kill_switch_rejects_everything():
    ks = _ks()
    ks.activate("manual")
    d = _evaluate([_buy(), _buy("MSFT")], _portfolio(), ks=ks)
    assert d.approved == []
    assert {r.reason for r in d.rejected} == {RejectReason.KILL_SWITCH_ACTIVE}


def test_cooldown_rejects_with_distinct_reason():
    ks = _ks()
    ks.start_cooldown("daily_loss_breaker", LIMITS)
    d = _evaluate([_buy()], _portfolio(), ks=ks)
    assert d.rejected[0].reason is RejectReason.BREAKER_COOLDOWN


def test_drawdown_activates_kill_switch_and_rejects():
    ks = _ks()
    p = _portfolio(
        equity=Decimal(84_000),
        day_start_equity=Decimal(84_000),
        week_start_equity=Decimal(84_000),
    )  # 16% below peak 100k
    d = _evaluate([_buy()], p, ks=ks)
    assert d.rejected[0].reason is RejectReason.MAX_DRAWDOWN_EXCEEDED
    halted, reason = ks.is_halted()
    assert halted and reason == "kill_switch:max_drawdown_exceeded"


def test_daily_breaker_arms_cooldown():
    ks = _ks()
    p = _portfolio(equity=Decimal(96_000))  # 4% daily loss
    d = _evaluate([_buy()], p, ks=ks)
    assert d.rejected[0].reason is RejectReason.DAILY_LOSS_BREAKER
    assert ks.is_halted()[1] == "cooldown:daily_loss_breaker"


def test_macro_pause_rejects_all():
    d = _evaluate([_buy()], _portfolio(), env=MarketEnvironment(macro_event_pause=True))
    assert d.rejected[0].reason is RejectReason.MACRO_EVENT_PAUSE


def test_vix_above_threshold_halves_buy_size():
    d = _evaluate([_buy()], _portfolio(), env=MarketEnvironment(vix=Decimal(35)))
    assert d.approved[0].quantity == Decimal(50)  # 100 halved
    assert d.audit.get("vix_halved") is True


def test_vix_at_threshold_does_not_halve():
    d = _evaluate([_buy()], _portfolio(), env=MarketEnvironment(vix=Decimal(30)))
    assert d.approved[0].quantity == Decimal(100)


def test_unknown_vix_does_not_halve():
    d = _evaluate([_buy()], _portfolio(), env=MarketEnvironment(vix=None))
    assert d.approved[0].quantity == Decimal(100)


def test_sell_without_position_rejected():
    sell = TradeProposal(
        symbol="AAPL", asset_class="stock", action="sell", confidence=0.9, reasoning="r"
    )
    d = _evaluate([sell], _portfolio())
    assert d.rejected[0].reason is RejectReason.NO_POSITION_TO_SELL


def test_sell_closes_full_position_with_no_stop():
    sell = TradeProposal(
        symbol="AAPL", asset_class="stock", action="sell", confidence=0.9, reasoning="r"
    )
    p = _portfolio(
        positions={
            "AAPL": Position(
                symbol="AAPL",
                asset_class="stock",
                quantity=Decimal("12.5"),
                avg_entry_price=Decimal(90),
            )
        }
    )
    d = _evaluate([sell], p)
    (order,) = d.approved
    assert order.side == "sell"
    assert order.quantity == Decimal("12.5")
    assert order.stop_loss_price is None


def test_insufficient_candles_rejects_not_crashes():
    d = _evaluate([_buy()], _portfolio(), candles={"AAPL": _candles(5)})
    assert d.rejected[0].reason is RejectReason.INSUFFICIENT_DATA


def test_missing_price_rejects():
    d = _evaluate([_buy()], _portfolio(), prices={})
    assert d.rejected[0].reason is RejectReason.INSUFFICIENT_DATA


def test_cash_at_reserve_floor_rejects_buy():
    p = _portfolio(cash=Decimal(20_000))  # exactly the 20% reserve
    d = _evaluate([_buy()], p)
    assert d.rejected[0].reason is RejectReason.INSUFFICIENT_CASH_AFTER_RESERVE


def test_mixed_batch_evaluated_independently():
    sell_no_pos = TradeProposal(
        symbol="MSFT", asset_class="stock", action="sell", confidence=0.7, reasoning="r"
    )
    d = _evaluate([_buy(), sell_no_pos], _portfolio())
    assert len(d.approved) == 1 and d.approved[0].symbol == "AAPL"
    assert len(d.rejected) == 1 and d.rejected[0].proposal.symbol == "MSFT"


def test_gate_never_reads_llm_confidence_for_sizing():
    """Same portfolio, wildly different confidences → identical quantity.
    Confidence is audit data, not a sizing input (RISK_RULES structural rule)."""
    low = _evaluate([_buy(confidence=0.01)], _portfolio()).approved[0].quantity
    high = _evaluate([_buy(confidence=0.99)], _portfolio()).approved[0].quantity
    assert low == high == Decimal(100)


def test_buy_into_untracked_exposure_rejected():
    """Held exposure whose size is unquantifiable this cycle: the gate must
    refuse to ADD to it (and surface it in the audit), never treat it as flat."""
    p = _portfolio(untracked_positions={"AAPL": Decimal(7)})
    d = _evaluate([_buy()], p)
    assert d.approved == []
    assert d.rejected[0].reason is RejectReason.INSUFFICIENT_DATA
    assert d.audit["untracked_positions"] == {"AAPL": "7"}


def test_untracked_exposure_does_not_block_other_symbols():
    p = _portfolio(untracked_positions={"XYZ-USD": Decimal(7)})
    d = _evaluate([_buy()], p)
    assert len(d.approved) == 1 and d.approved[0].symbol == "AAPL"


# --- liquidity filter (D-022) ------------------------------------------------


def _crypto_candles(volume: Decimal, n: int = 20) -> list[Candle]:
    """Crypto candles whose stored volume is USD already (CoinGecko convention)."""
    return [
        Candle(
            symbol="BTC-USD",
            asset_class="crypto",
            ts=T0 + timedelta(days=i),
            open=Decimal(60_000),
            high=Decimal(61_000),
            low=Decimal(59_000),
            close=Decimal(60_000),
            volume=volume,
            source="test",
        )
        for i in range(n)
    ]


def test_illiquid_buy_rejected_with_measured_figure():
    thin = _candles(volume=Decimal(1))  # $100/day
    d = _evaluate([_buy()], _portfolio(), candles={"AAPL": thin})
    assert d.approved == []
    (r,) = d.rejected
    assert r.reason is RejectReason.ILLIQUID
    assert r.detail == "20d avg dollar volume $100.00 vs stock minimum $10,000,000.00"


def test_illiquid_rejection_is_audited():
    thin = _candles(volume=Decimal(1))
    d = _evaluate([_buy()], _portfolio(), candles={"AAPL": thin})
    assert d.audit["illiquid"] == {
        "AAPL": "20d avg dollar volume $100.00 vs stock minimum $10,000,000.00"
    }


def test_liquid_symbols_produce_no_illiquid_audit_key():
    d = _evaluate([_buy()], _portfolio())
    assert "illiquid" not in d.audit


def test_buy_exactly_at_the_stock_threshold_is_approved():
    # close 100 x volume -> $10,000,000/day, exactly the minimum.
    at = _candles(volume=LIMITS.min_avg_dollar_volume_stock / Decimal(100))
    d = _evaluate([_buy()], _portfolio(), candles={"AAPL": at})
    assert d.rejected == [] and len(d.approved) == 1


def test_buy_one_cent_below_the_stock_threshold_is_rejected():
    below = (LIMITS.min_avg_dollar_volume_stock - Decimal("0.01")) / Decimal(100)
    d = _evaluate([_buy()], _portfolio(), candles={"AAPL": _candles(volume=below)})
    assert d.rejected[0].reason is RejectReason.ILLIQUID


def test_sells_are_never_liquidity_filtered():
    """Closing a position reduces risk - thin volume must not trap us in it."""
    sell = TradeProposal(
        symbol="AAPL", asset_class="stock", action="sell", confidence=0.9, reasoning="r"
    )
    p = _portfolio(
        positions={
            "AAPL": Position(
                symbol="AAPL",
                asset_class="stock",
                quantity=Decimal(5),
                avg_entry_price=Decimal(90),
            )
        }
    )
    d = _evaluate([sell], p, candles={"AAPL": _candles(volume=Decimal(1))})
    assert d.rejected == []
    assert d.approved[0].side == "sell" and d.approved[0].quantity == Decimal(5)


def test_too_few_bars_is_insufficient_data_not_illiquid():
    """19 liquid sessions cannot be measured over a 20-day window."""
    d = _evaluate([_buy()], _portfolio(), candles={"AAPL": _candles(n=19)})
    (r,) = d.rejected
    assert r.reason is RejectReason.INSUFFICIENT_DATA
    # Pins WHICH check fired: 19 bars clears ATR(14) but not the 20-day window.
    assert r.detail == "liquidity needs at least 20 distinct-timestamp candles, got 19"


def test_crypto_uses_the_crypto_threshold_derived_from_the_candles():
    """$20M/day clears the stock minimum but not the crypto one; the gate must
    take the class from the candles, not from the (lying) proposal."""
    proposal = TradeProposal(
        symbol="BTC-USD", asset_class="stock", action="buy", confidence=0.8, reasoning="r"
    )
    d = evaluate(
        [proposal],
        _portfolio(),
        {"BTC-USD": _crypto_candles(Decimal(20_000_000))},
        {"BTC-USD": Decimal(60_000)},
        ENV_OK,
        _ks(),
        LIMITS,
    )
    assert d.rejected[0].reason is RejectReason.ILLIQUID


def test_crypto_volume_is_not_multiplied_by_close_at_the_gate():
    """$100/day of crypto volume must reject; close x volume would read $6M."""
    proposal = TradeProposal(
        symbol="BTC-USD", asset_class="crypto", action="buy", confidence=0.8, reasoning="r"
    )
    d = evaluate(
        [proposal],
        _portfolio(),
        {"BTC-USD": _crypto_candles(Decimal(100))},
        {"BTC-USD": Decimal(60_000)},
        ENV_OK,
        _ks(),
        LIMITS,
    )
    assert d.rejected[0].reason is RejectReason.ILLIQUID
    assert d.rejected[0].detail == (
        "20d avg dollar volume $100.00 vs crypto minimum $50,000,000.00"
    )


def test_liquid_crypto_is_approved():
    proposal = TradeProposal(
        symbol="BTC-USD", asset_class="crypto", action="buy", confidence=0.8, reasoning="r"
    )
    d = evaluate(
        [proposal],
        _portfolio(),
        {"BTC-USD": _crypto_candles(Decimal(30_000_000_000))},
        {"BTC-USD": Decimal(60_000)},
        ENV_OK,
        _ks(),
        LIMITS,
    )
    assert d.rejected == [] and d.approved[0].asset_class == "crypto"


def test_illiquid_symbol_does_not_block_a_liquid_one():
    d = evaluate(
        [_buy("THIN"), _buy("AAPL")],
        _portfolio(),
        {"THIN": _candles(volume=Decimal(1), symbol="THIN"), "AAPL": _candles()},
        {"THIN": Decimal(100), "AAPL": Decimal(100)},
        ENV_OK,
        _ks(),
        LIMITS,
    )
    assert [o.symbol for o in d.approved] == ["AAPL"]
    assert d.rejected[0].reason is RejectReason.ILLIQUID
