"""Portfolio tracker tests: pure helpers with synthetic data, tracker with
fake brokers and an in-memory-ish fake repo layer (no DB, no network)."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError
from structlog.testing import capture_logs

from trading_agent.brokers.base import AccountSummary, BrokerPosition
from trading_agent.data.models import Candle
from trading_agent.portfolio import tracker as tracker_mod
from trading_agent.portfolio.tracker import (
    PortfolioTracker,
    average_cost_walk,
    day_start,
    week_start,
)
from trading_agent.risk.limits import RiskLimits
from trading_agent.risk.models import PortfolioState, Position
from trading_agent.storage.models import CashFlow

# The live limits have no trail (D-032); these tests pin the trail machinery at 3%.
LIMITS = RiskLimits(trailing_stop_pct=Decimal("0.03"))
NOW = datetime(2026, 9, 12, 15, 30, tzinfo=UTC)  # a Saturday


def _fill(side: str, qty: str, price: str, at: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        side=side, quantity=Decimal(qty), fill_price=Decimal(price), filled_at=at
    )


# --- pure helpers ------------------------------------------------------------


def test_average_cost_walk_weights_buys():
    fills = [
        _fill("buy", "10", "100", NOW - timedelta(days=5)),
        _fill("buy", "10", "110", NOW - timedelta(days=4)),
    ]
    qty, avg, entry_time = average_cost_walk(fills)
    assert qty == Decimal(20)
    assert avg == Decimal(105)
    assert entry_time == NOW - timedelta(days=5)


def test_average_cost_walk_sell_keeps_avg():
    fills = [
        _fill("buy", "10", "100", NOW - timedelta(days=5)),
        _fill("sell", "4", "120", NOW - timedelta(days=3)),
    ]
    qty, avg, _ = average_cost_walk(fills)
    assert qty == Decimal(6)
    assert avg == Decimal(100)


def test_average_cost_walk_flat_resets_basis():
    fills = [
        _fill("buy", "10", "100", NOW - timedelta(days=9)),
        _fill("sell", "10", "120", NOW - timedelta(days=8)),
        _fill("buy", "5", "200", NOW - timedelta(days=2)),
    ]
    qty, avg, entry_time = average_cost_walk(fills)
    assert qty == Decimal(5)
    assert avg == Decimal(200)  # old basis gone
    assert entry_time == NOW - timedelta(days=2)


def test_average_cost_walk_empty_and_flat():
    assert average_cost_walk([]) == (Decimal(0), None, None)
    fills = [
        _fill("buy", "10", "100", NOW - timedelta(days=2)),
        _fill("sell", "10", "90", NOW - timedelta(days=1)),
    ]
    assert average_cost_walk(fills) == (Decimal(0), None, None)


def test_day_and_week_start_utc():
    assert day_start(NOW) == datetime(2026, 9, 12, tzinfo=UTC)
    # 2026-09-12 is a Saturday; ISO week began Monday 2026-09-07.
    assert week_start(NOW) == datetime(2026, 9, 7, tzinfo=UTC)


def test_week_start_on_monday_is_same_day():
    monday = datetime(2026, 9, 7, 8, 0, tzinfo=UTC)
    assert week_start(monday) == datetime(2026, 9, 7, tzinfo=UTC)


# --- tracker with fakes --------------------------------------------------------


class FakeTradeRepo:
    fills: ClassVar[dict[str, list]] = {}
    streak: ClassVar[int] = 0
    seen: ClassVar[dict] = {}

    def __init__(self, _session) -> None: ...

    def fills_for_symbol(self, symbol, trading_mode, since=None):
        FakeTradeRepo.seen["fills_mode"] = trading_mode
        FakeTradeRepo.seen["fills_since"] = since
        return [
            f for f in FakeTradeRepo.fills.get(symbol, []) if since is None or f.filled_at >= since
        ]

    def venue_fills(self, asset_class, trading_mode, since):
        fills = [
            (s, f) for s, fs in FakeTradeRepo.fills.items() for f in fs if f.filled_at >= since
        ]
        return [
            SimpleNamespace(symbol=s, **vars(f))
            for s, f in sorted(fills, key=lambda sf: sf[1].filled_at)
        ]

    def trailing_loss_streak(self, trading_mode, since):
        FakeTradeRepo.seen["streak_mode"] = trading_mode
        FakeTradeRepo.seen["streak_since"] = since
        return FakeTradeRepo.streak


class FakeSnapshotRepo:
    anchors: ClassVar[dict] = {}  # {boundary datetime: equity}; the snapshot stands AT the boundary
    peak: ClassVar[Decimal | None] = None
    peak_at: ClassVar[datetime] = NOW - timedelta(hours=1)
    recorded: ClassVar[list] = []
    seen: ClassVar[dict] = {}

    def __init__(self, _session) -> None: ...

    def snapshot_at(self, boundary, trading_mode, since=None):
        FakeSnapshotRepo.seen["mode"] = trading_mode
        FakeSnapshotRepo.seen["since"] = since
        equity = FakeSnapshotRepo.anchors.get(boundary)
        return None if equity is None else SimpleNamespace(ts=boundary, equity=equity)

    def latest(self, trading_mode, since=None):
        FakeSnapshotRepo.seen["latest_since"] = since
        peak = FakeSnapshotRepo.peak
        return (
            None if peak is None else SimpleNamespace(ts=FakeSnapshotRepo.peak_at, peak_equity=peak)
        )

    def record(self, **kw):
        FakeSnapshotRepo.recorded.append(kw)


class FakeCashFlowRepo:
    rows: ClassVar[list] = []  # the ledger; add() "commits" straight into it

    def __init__(self, _session) -> None: ...

    def ledger(self, trading_mode):
        return list(FakeCashFlowRepo.rows)

    def start(self, trading_mode):
        return max((r.ts for r in FakeCashFlowRepo.rows if r.kind == "baseline"), default=None)

    def add(self, rows):
        FakeCashFlowRepo.rows.extend(rows)


def _use_fakes(monkeypatch):
    FakeTradeRepo.fills = {}
    FakeTradeRepo.streak = 0
    FakeTradeRepo.seen = {}
    FakeSnapshotRepo.anchors = {}
    FakeSnapshotRepo.peak = None
    FakeSnapshotRepo.recorded = []
    FakeSnapshotRepo.seen = {}
    FakeSnapshotRepo.peak_at = NOW - timedelta(hours=1)
    FakeCashFlowRepo.rows = []
    monkeypatch.setattr(tracker_mod, "CashFlowRepository", FakeCashFlowRepo)
    monkeypatch.setattr(tracker_mod, "TradeRepository", FakeTradeRepo)
    monkeypatch.setattr(tracker_mod, "SnapshotRepository", FakeSnapshotRepo)


def _tracker(brokers) -> PortfolioTracker:
    return PortfolioTracker(_session_factory(), brokers, trading_mode="paper", limits=LIMITS)


def _session_factory():
    factory = MagicMock()
    factory.return_value.__enter__.return_value = MagicMock()
    return factory


def _broker(
    equity: str, cash: str, positions: list[BrokerPosition], asset_class: str = "stock"
) -> MagicMock:
    broker = MagicMock()
    broker.asset_class = asset_class
    broker.get_account.return_value = AccountSummary(equity=Decimal(equity), cash=Decimal(cash))
    broker.get_positions.return_value = positions
    return broker


def test_build_state_sums_brokers_and_uses_reported_entries(monkeypatch):
    _use_fakes(monkeypatch)
    FakeSnapshotRepo.anchors = {
        day_start(NOW): Decimal(99_000),
        week_start(NOW): Decimal(95_000),
    }
    FakeSnapshotRepo.peak = Decimal(120_000)
    FakeTradeRepo.streak = 3

    alpaca = _broker(
        "60000",
        "40000",
        [
            BrokerPosition(
                symbol="AAPL",
                asset_class="stock",
                quantity=Decimal(50),
                avg_entry_price=Decimal(100),
            )
        ],
    )
    coinbase = _broker("40000", "10000", [], asset_class="crypto")
    state = _tracker([alpaca, coinbase]).build_state({}, now=NOW)

    assert state.equity == Decimal(100_000)
    assert state.cash == Decimal(50_000)
    # The gate spends each venue's own cash (D-032): pooled 50k, crypto venue 10k.
    assert state.venue_cash == {"stock": Decimal(40_000), "crypto": Decimal(10_000)}
    assert state.day_start_equity == Decimal(99_000)
    assert state.week_start_equity == Decimal(95_000)
    assert state.peak_equity == Decimal(120_000)  # prior peak higher than current
    assert state.consecutive_losses == 3
    assert state.positions["AAPL"].avg_entry_price == Decimal(100)
    # All history queries mode-scoped; streak window bounded.
    assert FakeSnapshotRepo.seen["mode"] == "paper"
    assert FakeTradeRepo.seen["streak_mode"] == "paper"
    assert FakeTradeRepo.seen["streak_since"] == NOW - timedelta(days=7)


def test_build_state_derives_entry_from_trade_history(monkeypatch):
    """Coinbase reports balances (no lots) — the entry must come from fills."""
    _use_fakes(monkeypatch)
    FakeTradeRepo.fills = {
        "BTC-USD": [
            _fill("buy", "0.3", "90000", NOW - timedelta(days=6)),
            _fill("buy", "0.2", "95000", NOW - timedelta(days=2)),
        ]
    }
    coinbase = _broker(
        "88500",
        "40000",
        [
            BrokerPosition(
                symbol="BTC-USD",
                asset_class="crypto",
                quantity=Decimal("0.5"),
                avg_entry_price=None,
            )
        ],
    )
    state = _tracker([coinbase]).build_state({}, now=NOW)
    assert state.positions["BTC-USD"].avg_entry_price == Decimal(92000)


def test_build_state_fresh_day_anchors_at_current_equity(monkeypatch):
    _use_fakes(monkeypatch)  # no snapshots at all
    broker = _broker("100000", "100000", [])
    state = _tracker([broker]).build_state({}, now=NOW)
    assert state.day_start_equity == Decimal(100_000)
    assert state.week_start_equity == Decimal(100_000)
    assert state.peak_equity == Decimal(100_000)
    assert state.consecutive_losses == 0


def test_build_state_current_equity_can_be_new_peak(monkeypatch):
    _use_fakes(monkeypatch)
    FakeSnapshotRepo.peak = Decimal(90_000)
    broker = _broker("100000", "100000", [])
    state = _tracker([broker]).build_state({}, now=NOW)
    assert state.peak_equity == Decimal(100_000)


def test_build_state_entry_falls_back_to_price_with_no_history(monkeypatch):
    _use_fakes(monkeypatch)
    coinbase = _broker(
        "97000",
        "0",
        [
            BrokerPosition(
                symbol="ETH-USD",
                asset_class="crypto",
                quantity=Decimal(10),
                avg_entry_price=None,
            )
        ],
    )
    state = _tracker([coinbase]).build_state({"ETH-USD": Decimal(3000)}, now=NOW)
    assert state.positions["ETH-USD"].avg_entry_price == Decimal(3000)


def test_unpriceable_position_becomes_untracked_not_lost(monkeypatch):
    """Exposure with no entry and no price stays VISIBLE to the gate as untracked -
    it must never silently vanish from the risk picture."""
    _use_fakes(monkeypatch)
    coinbase = _broker(
        "97000",
        "0",
        [
            BrokerPosition(
                symbol="XYZ-USD",
                asset_class="crypto",
                quantity=Decimal(7),
                avg_entry_price=None,
            )
        ],
    )
    state = _tracker([coinbase]).build_state({}, now=NOW)
    assert "XYZ-USD" not in state.positions
    assert state.untracked_positions == {"XYZ-USD": Decimal(7)}


def test_snapshot_persists_serializable_positions(monkeypatch):
    _use_fakes(monkeypatch)
    state = PortfolioState(
        equity=Decimal(100_000),
        cash=Decimal(50_000),
        peak_equity=Decimal(100_000),
        day_start_equity=Decimal(100_000),
        week_start_equity=Decimal(100_000),
        consecutive_losses=0,
        positions={
            "AAPL": Position(
                symbol="AAPL",
                asset_class="stock",
                quantity=Decimal(50),
                avg_entry_price=Decimal(100),
            )
        },
    )
    _tracker([]).snapshot(state, now=NOW)
    (row,) = FakeSnapshotRepo.recorded
    assert row["equity"] == Decimal(100_000)
    assert row["positions"] == {"AAPL": {"qty": "50", "avg_entry": "100"}}
    assert row["trading_mode"] == "paper"


# --- trailing stops ---------------------------------------------------------------


def _candles(symbol: str, highs: list[str], start: datetime) -> list[Candle]:
    return [
        Candle(
            symbol=symbol,
            asset_class="stock",
            ts=start + timedelta(days=i),
            open=Decimal(h) - 1,
            high=Decimal(h),
            low=Decimal(h) - 2,
            close=Decimal(h) - 1,
            volume=Decimal(1),
            source="test",
        )
        for i, h in enumerate(highs)
    ]


def _stock_state(entry: str = "100", qty: str = "50") -> PortfolioState:
    return PortfolioState(
        equity=Decimal(100_000),
        cash=Decimal(50_000),
        peak_equity=Decimal(100_000),
        day_start_equity=Decimal(100_000),
        week_start_equity=Decimal(100_000),
        consecutive_losses=0,
        positions={
            "AAPL": Position(
                symbol="AAPL",
                asset_class="stock",
                quantity=Decimal(qty),
                avg_entry_price=Decimal(entry),
            )
        },
    )


def test_trailing_stop_breach_detected_after_runup(monkeypatch):
    """Price ran to 120 then fell to 115: the 3% trail off 120 (116.4) is hit."""
    _use_fakes(monkeypatch)
    entry_time = NOW - timedelta(days=20)
    FakeTradeRepo.fills = {"AAPL": [_fill("buy", "50", "100", entry_time)]}
    highs = ["101"] * 16 + ["110", "115", "120", "118"]
    candles = _candles("AAPL", highs, entry_time)

    breaches = _tracker([]).check_trailing_stops(
        _stock_state(), {"AAPL": candles}, {"AAPL": Decimal(115)}
    )
    (breach,) = breaches
    assert breach.symbol == "AAPL"
    assert breach.quantity == Decimal(50)
    assert breach.stop_price == Decimal("116.40")  # 120 x 0.97
    assert breach.current_price == Decimal(115)


def test_no_breach_when_price_above_stop(monkeypatch):
    _use_fakes(monkeypatch)
    entry_time = NOW - timedelta(days=20)
    FakeTradeRepo.fills = {"AAPL": [_fill("buy", "50", "100", entry_time)]}
    candles = _candles("AAPL", ["101"] * 20, entry_time)
    breaches = _tracker([]).check_trailing_stops(
        _stock_state(), {"AAPL": candles}, {"AAPL": Decimal(100)}
    )
    assert breaches == []  # trail off 101 = 97.97, price 100 above it


def test_high_water_ignores_candles_before_entry(monkeypatch):
    """A spike BEFORE this position's entry must not ratchet its trail."""
    _use_fakes(monkeypatch)
    entry_time = NOW - timedelta(days=5)
    FakeTradeRepo.fills = {"AAPL": [_fill("buy", "50", "100", entry_time)]}
    pre_spike = _candles("AAPL", ["200"] * 15, NOW - timedelta(days=25))
    post_entry = _candles("AAPL", ["101"] * 5, entry_time)
    breaches = _tracker([]).check_trailing_stops(
        _stock_state(), {"AAPL": pre_spike + post_entry}, {"AAPL": Decimal(100)}
    )
    assert breaches == []  # trail anchors at 101, not the pre-entry 200


def test_no_fill_history_skips_never_liquidates(monkeypatch):
    """An Alpaca position with broker-reported entry but NO fill rows (a fresh DB)
    must be SKIPPED. Widening the high-water window to all candles would flag any
    position bought below a prior high for full liquidation."""
    _use_fakes(monkeypatch)  # fills empty
    candles = _candles("AAPL", ["200"] * 15 + ["101"] * 5, NOW - timedelta(days=25))
    breaches = _tracker([]).check_trailing_stops(
        _stock_state(), {"AAPL": candles}, {"AAPL": Decimal(100)}
    )
    assert breaches == []  # 0.97 x 200 = 194 would have "breached" at price 100


def test_entry_day_candle_counts_toward_high_water(monkeypatch):
    """Daily bars are stamped 00:00 UTC; an intraday entry must still let the
    entry day's own bar ratchet the trail."""
    _use_fakes(monkeypatch)
    entry_time = day_start(NOW) - timedelta(days=1) + timedelta(hours=15)  # yesterday 15:00
    FakeTradeRepo.fills = {"AAPL": [_fill("buy", "50", "100", entry_time)]}
    # Entry day's bar (stamped at its 00:00) ran to 120.
    candles = _candles("AAPL", ["120", "118"], day_start(entry_time))
    breaches = _tracker([]).check_trailing_stops(
        _stock_state(), {"AAPL": candles}, {"AAPL": Decimal(115)}
    )
    (breach,) = breaches
    assert breach.stop_price == Decimal("116.40")  # trail off the entry-day 120


def test_missing_price_skipped_not_guessed(monkeypatch):
    """No current price: neither 'safe' nor 'breached' - skipped with warning."""
    _use_fakes(monkeypatch)
    FakeTradeRepo.fills = {"AAPL": [_fill("buy", "50", "100", NOW - timedelta(days=5))]}
    breaches = _tracker([]).check_trailing_stops(
        _stock_state(), {"AAPL": _candles("AAPL", ["101"] * 5, NOW - timedelta(days=5))}, {}
    )
    assert breaches == []


def test_live_limits_detect_no_trail_breach(monkeypatch):
    """D-032: no fixed trail - a run-up and a 4% pullback breach nothing; the
    venue ATR stop is the only exit rule."""
    _use_fakes(monkeypatch)
    entry_time = NOW - timedelta(days=20)
    FakeTradeRepo.fills = {"AAPL": [_fill("buy", "50", "100", entry_time)]}
    candles = _candles("AAPL", ["101"] * 16 + ["110", "115", "120", "118"], entry_time)
    tracker = PortfolioTracker(_session_factory(), [], trading_mode="paper")
    state = _stock_state()
    assert tracker.check_trailing_stops(state, {"AAPL": candles}, {"AAPL": Decimal(115)}) == []


# --- the paper ledger (D-040) --------------------------------------------------


def _flow(kind, venue, amount, at, equity_before="0", scheduled_for=None):
    """An unsaved ledger row."""
    return CashFlow(
        ts=at,
        trading_mode="paper",
        asset_class=venue,
        kind=kind,
        amount=Decimal(amount),
        equity_before=Decimal(equity_before),
        scheduled_for=scheduled_for,
    )


def _started(at: datetime) -> list[CashFlow]:
    """A ledger started at `at` over $100k stock / $10k crypto paper venues.

    Baselines, then the first $1,000 (900 stock / 100 crypto).
    """
    return [
        _flow("baseline", "stock", "-100000", at),
        _flow("baseline", "crypto", "-10000", at),
        _flow("deposit", "crypto", "100", at, "0", at.date()),
        _flow("deposit", "stock", "900", at, "100", at.date()),
    ]


def test_the_ledger_is_all_the_app_sees_of_a_paper_venue(monkeypatch):
    _use_fakes(monkeypatch)
    start = NOW - timedelta(days=3)
    FakeCashFlowRepo.rows = _started(start)
    stock = _broker("100000", "100000", [])
    crypto = _broker("10000", "10000", [], asset_class="crypto")
    state = _tracker([stock, crypto]).build_state({}, now=NOW)
    assert state.equity == Decimal(1000)
    assert state.cash == Decimal(1000)
    assert state.venue_cash == {"stock": Decimal(900), "crypto": Decimal(100)}
    # History from before the ledger started is ignored: anchors, peak and loss streak.
    assert FakeSnapshotRepo.seen["since"] == start
    assert FakeSnapshotRepo.seen["latest_since"] == start
    assert FakeTradeRepo.seen["streak_since"] == start


def test_a_due_deposit_is_booked_and_moves_no_anchor(monkeypatch):
    """Day 14 after the start books $500 (450 stock / 50 crypto); the day, week and
    peak anchors all grow by the deposit's share, so the gate sees no gain."""
    _use_fakes(monkeypatch)
    start = NOW - timedelta(days=14, hours=1)
    FakeCashFlowRepo.rows = _started(start)
    FakeSnapshotRepo.anchors = {day_start(NOW): Decimal(1000), week_start(NOW): Decimal(1000)}
    FakeSnapshotRepo.peak = Decimal(1000)
    stock = _broker("100000", "100000", [])
    crypto = _broker("10000", "10000", [], asset_class="crypto")
    # /status only reads: nothing is booked, and its equity is the pre-deposit one.
    assert _tracker([stock, crypto]).build_state({}, now=NOW).equity == Decimal(1000)
    assert len(FakeCashFlowRepo.rows) == 4
    state = _tracker([stock, crypto]).build_state({}, now=NOW, book_deposits=True)
    booked = FakeCashFlowRepo.rows[4:]
    assert [(f.asset_class, f.amount, f.equity_before) for f in booked] == [
        ("crypto", Decimal(50), Decimal(1000)),
        ("stock", Decimal(450), Decimal(1050)),
    ]
    assert all(f.scheduled_for == (start + timedelta(days=14)).date() for f in booked)
    assert state.equity == Decimal(1500)
    assert state.venue_cash == {"stock": Decimal(1350), "crypto": Decimal(150)}
    assert state.day_start_equity == state.week_start_equity == state.peak_equity == Decimal(1500)
    # The next cycle books nothing more.
    _tracker([stock, crypto]).build_state({}, now=NOW + timedelta(minutes=15), book_deposits=True)
    assert len(FakeCashFlowRepo.rows) == 6


def test_two_overdue_deposits_chain_their_equity_before(monkeypatch):
    """Agent off for a month: both dates book at once, and together they scale the
    anchors by exactly 2x ($1,000 -> $2,000), not 2.25x."""
    _use_fakes(monkeypatch)
    start = NOW - timedelta(days=28, hours=1)
    FakeCashFlowRepo.rows = _started(start)
    FakeSnapshotRepo.peak = Decimal(1000)
    stock = _broker("100000", "100000", [])
    crypto = _broker("10000", "10000", [], asset_class="crypto")
    state = _tracker([stock, crypto]).build_state({}, now=NOW, book_deposits=True)
    assert [f.equity_before for f in FakeCashFlowRepo.rows[4:]] == [
        Decimal(1000),
        Decimal(1050),
        Decimal(1500),
        Decimal(1550),
    ]
    assert state.equity == state.peak_equity == Decimal(2000)


def test_a_date_booked_elsewhere_rolls_back_and_rereads(monkeypatch):
    _use_fakes(monkeypatch)
    FakeCashFlowRepo.rows = _started(NOW - timedelta(days=14, hours=1))
    session = MagicMock()
    session.commit.side_effect = IntegrityError("insert", {}, Exception("duplicate"))
    factory = MagicMock()
    factory.return_value.__enter__.return_value = session
    tracker = PortfolioTracker(factory, [_broker("100000", "100000", [])], "paper", LIMITS)
    FakeCashFlowRepo.rows = [f for f in FakeCashFlowRepo.rows if f.asset_class == "stock"]
    real_add = FakeCashFlowRepo.add
    monkeypatch.setattr(FakeCashFlowRepo, "add", lambda self, rows: None)  # the insert failed
    state = tracker.build_state({}, now=NOW, book_deposits=True)
    monkeypatch.setattr(FakeCashFlowRepo, "add", real_add)
    session.rollback.assert_called_once()
    assert state.equity == Decimal(900)  # the ledger as it stands: nothing booked


def test_a_venue_outside_the_ledger_fails_closed(monkeypatch):
    """Alpaca keys added after a crypto-only start: $100k of play money must never
    become the app's equity."""
    _use_fakes(monkeypatch)
    FakeCashFlowRepo.rows = [
        f for f in _started(NOW - timedelta(days=1)) if f.asset_class == "crypto"
    ]
    brokers = [_broker("100000", "100000", []), _broker("10000", "10000", [], "crypto")]
    with pytest.raises(ValueError, match="restart the ledger"):
        _tracker(brokers).build_state({}, now=NOW)


def test_a_deposit_does_not_dilute_a_loss_already_taken(monkeypatch):
    """Down 10% from the peak when $500 lands: still down 10% after it."""
    _use_fakes(monkeypatch)
    start = NOW - timedelta(days=14, hours=1)
    day14 = (start + timedelta(days=14)).date()
    FakeCashFlowRepo.rows = [
        *(f for f in _started(start) if f.asset_class == "stock"),
        _flow("deposit", "stock", "500", NOW - timedelta(minutes=30), "900", day14),
    ]
    FakeSnapshotRepo.peak = Decimal(1000)
    FakeSnapshotRepo.peak_at = NOW - timedelta(hours=1)
    stock = _broker("100000", "100000", [])  # the app's view: 900 left + the 500 deposit
    state = _tracker([stock]).build_state({}, now=NOW, book_deposits=True)
    assert state.equity == Decimal(1400)
    assert 1 - state.equity / state.peak_equity == pytest.approx(Decimal("0.1"))


# --- app-held swing stops (D-054) -----------------------------------------------------


def _swing_state(*symbols: str) -> PortfolioState:
    """A portfolio holding 0.18 shares of each symbol."""
    return PortfolioState(
        equity=Decimal(1000),
        cash=Decimal(500),
        peak_equity=Decimal(1000),
        day_start_equity=Decimal(1000),
        week_start_equity=Decimal(1000),
        consecutive_losses=0,
        positions={
            s: Position(
                symbol=s,
                asset_class="stock",
                quantity=Decimal("0.18"),
                avg_entry_price=Decimal(500),
            )
            for s in symbols
        },
    )


def test_app_held_stops_breach_at_or_below_and_skip_what_they_cannot_judge(monkeypatch):
    stops = {"AAPL": Decimal(494), "MSFT": Decimal(494), "NVDA": Decimal(494), "XOM": None}

    class FakeOrders:
        def __init__(self, session):
            """Bind nothing."""

        def entry_stop(self, symbol, mode):
            """The stop stored on the symbol's buy."""
            return stops[symbol]

    monkeypatch.setattr(tracker_mod, "OrderRepository", FakeOrders)
    prices = {"AAPL": Decimal(494), "MSFT": Decimal("494.01"), "XOM": Decimal(1)}
    state = _swing_state(*stops)
    tracker = _tracker([])

    with capture_logs() as logs:
        breaches = tracker.check_swing_stops(state, prices, set(stops))
        tracker.check_swing_stops(state, prices, set(stops))

    assert [(b.symbol, b.stop_price, b.quantity) for b in breaches] == [
        ("AAPL", Decimal(494), Decimal("0.18"))
    ]
    events = [(e["log_level"], e["event"], e["symbol"]) for e in logs if "symbol" in e]
    # No price (NVDA): skipped with a warning, never called safe or breached.
    assert ("warning", "swing_stop_unchecked", "NVDA") in events
    # No stored stop (XOM): an error the first time, a warning after.
    xom = [level for level, _, symbol in events if symbol == "XOM"]
    assert xom == ["error", "warning"]


def test_the_missing_stop_error_returns_after_the_symbol_was_gone(monkeypatch):
    class FakeOrders:
        def __init__(self, session):
            """Bind nothing."""

        def entry_stop(self, symbol, mode):
            """No stored stop."""
            return None

    monkeypatch.setattr(tracker_mod, "OrderRepository", FakeOrders)
    tracker, prices = _tracker([]), {"XOM": Decimal(1)}
    with capture_logs() as logs:
        tracker.check_swing_stops(_swing_state("XOM"), prices, {"XOM"})
        tracker.check_swing_stops(_swing_state(), prices, set())  # sold
        tracker.check_swing_stops(_swing_state("XOM"), prices, {"XOM"})  # bought again
    levels = [e["log_level"] for e in logs if e.get("symbol") == "XOM"]
    assert levels == ["error", "error"]


# --- the broker as referee --------------------------------------------------------


def _referee(monkeypatch, **fills):
    """A tracker over a ledger started a day ago, with symbol -> fills booked."""
    _use_fakes(monkeypatch)
    FakeCashFlowRepo.rows = _started(NOW - timedelta(days=1))
    FakeTradeRepo.fills = fills
    return _tracker([_broker("1000", "500", [])])


def _mismatch_events(logs):
    """(event, level, symbol) for every position-check log line."""
    return [
        (e["event"], e["log_level"], e["symbol"])
        for e in logs
        if e["event"].startswith("position_mismatch")
    ]


def test_a_lot_from_before_the_ledger_start_is_not_an_open_position(monkeypatch):
    tracker = _referee(
        monkeypatch,
        AAPL=[
            _fill("buy", "32", "200", NOW - timedelta(days=30)),  # sold at the cutover, unbooked
            _fill("buy", "0.18", "500", NOW - timedelta(hours=2)),
        ],
    )
    with capture_logs() as logs:
        assert tracker.check_positions(_swing_state("AAPL")) == {}
        assert tracker.check_positions(_swing_state("AAPL")) == {}
    assert _mismatch_events(logs) == []


def test_a_mismatch_pages_on_its_second_cycle_once_until_it_clears(monkeypatch):
    tracker = _referee(monkeypatch, AAPL=[_fill("buy", "1.18", "500", NOW - timedelta(hours=2))])
    held = _swing_state("AAPL")  # the broker holds 0.18
    with capture_logs() as logs:
        assert tracker.check_positions(held) == {"AAPL": (Decimal("1.18"), Decimal("0.18"))}
        tracker.check_positions(held)
        tracker.check_positions(held)
    assert _mismatch_events(logs) == [("position_mismatch", "error", "AAPL")]

    FakeTradeRepo.fills["AAPL"].append(_fill("sell", "1", "500", NOW - timedelta(hours=1)))
    with capture_logs() as logs:
        assert tracker.check_positions(held) == {}
    assert _mismatch_events(logs) == [("position_mismatch_cleared", "info", "AAPL")]

    FakeTradeRepo.fills["AAPL"].append(_fill("buy", "1", "500", NOW - timedelta(minutes=30)))
    with capture_logs() as logs:
        tracker.check_positions(held)
        tracker.check_positions(held)
    assert _mismatch_events(logs) == [("position_mismatch", "error", "AAPL")]


def test_a_one_cycle_mismatch_never_pages(monkeypatch):
    tracker = _referee(monkeypatch, AAPL=[_fill("buy", "0.18", "500", NOW - timedelta(hours=2))])
    with capture_logs() as logs:
        tracker.check_positions(_swing_state())  # the sale filled before its booking
        tracker.check_positions(_swing_state("AAPL"))
    assert _mismatch_events(logs) == []


def test_a_position_only_the_app_holds_is_a_mismatch(monkeypatch):
    tracker = _referee(monkeypatch, MSFT=[_fill("buy", "0.04", "500", NOW - timedelta(hours=2))])
    assert tracker.check_positions(_swing_state()) == {"MSFT": (Decimal("0.04"), Decimal(0))}


def test_rounding_below_the_tolerance_is_a_match(monkeypatch):
    tracker = _referee(
        monkeypatch, AAPL=[_fill("buy", "0.1800001", "500", NOW - timedelta(hours=2))]
    )
    assert tracker.check_positions(_swing_state("AAPL")) == {}


def test_without_a_ledger_nothing_is_compared(monkeypatch):
    tracker = _referee(monkeypatch, AAPL=[_fill("buy", "5", "500", NOW - timedelta(hours=2))])
    FakeCashFlowRepo.rows = []
    assert tracker.check_positions(_swing_state()) == {}
