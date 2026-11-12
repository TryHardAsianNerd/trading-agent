"""Full-cycle graph tests against the real compose TimescaleDB.

Skipped automatically when the database is unreachable (docker compose not
up). Everything else is faked: brokers, providers, Redis. Tests never touch
production-mode rows — unique symbols (T-*), source/broker "test", trading
modes "t-*", cycle ids "test-*" — and the module fixture cleans all of it up.
"""

import contextlib
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from structlog.testing import capture_logs
from tests.fakes import FakeRedis

from trading_agent.brokers.base import (
    AccountSummary,
    BrokerError,
    BrokerPosition,
    SubmittedOrder,
    VenueOrder,
)
from trading_agent.config import Settings
from trading_agent.data.models import Candle, Quote
from trading_agent.data.providers.base import ProviderError
from trading_agent.graph.builder import build_graph, postgres_checkpointer, run_cycle
from trading_agent.graph.deps import CycleDeps
from trading_agent.graph.state import TradeProposal
from trading_agent.portfolio.tracker import PortfolioTracker
from trading_agent.risk import macro_calendar
from trading_agent.risk.killswitch import (
    COOLDOWN_KEY,
    KILL_SWITCH_KEY,
    MANUAL_KILL_SWITCH_REASON,
    KillSwitch,
)
from trading_agent.risk.limits import DEFAULT_LIMITS, RiskLimits
from trading_agent.storage.db import (
    init_db,
    make_engine,
    make_session_factory,
    prune_checkpoints,
)
from trading_agent.storage.models import CashFlow, Decision, Order, PortfolioSnapshot
from trading_agent.storage.repos import OrderRepository, SnapshotRepository, TradeRepository


def _engine_or_skip():
    settings = Settings(_env_file=None)
    engine = make_engine(settings.database_url, connect_args={"connect_timeout": 3})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("TimescaleDB not reachable — run: docker compose up -d")
    return engine


@pytest.fixture(scope="module")
def engine():
    eng = _engine_or_skip()
    init_db(eng)
    yield eng
    with eng.begin() as conn:
        conn.execute(text("DELETE FROM ohlcv WHERE source IN ('test', 'itest')"))
        conn.execute(text("DELETE FROM trades WHERE broker = 'test'"))
        conn.execute(text("DELETE FROM orders WHERE broker = 'test'"))
        conn.execute(text("DELETE FROM portfolio_snapshots WHERE trading_mode LIKE 't-%'"))
        conn.execute(text("DELETE FROM cash_flows WHERE trading_mode LIKE 't-%'"))
        conn.execute(text("DELETE FROM decisions WHERE cycle_id LIKE 'test-%'"))
    for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
        # Own transaction + suppress per table: a failed shared txn would undo the deletes above.
        with contextlib.suppress(Exception), eng.begin() as conn:
            conn.execute(text(f"DELETE FROM {table} WHERE thread_id LIKE 'test-%'"))


@pytest.fixture()
def session_factory(engine):
    return make_session_factory(engine)


@pytest.fixture(autouse=True)
def _no_macro_events(monkeypatch):
    """Run every cycle test against an EMPTY macro calendar by default.

    The risk node reads the shipped calendar at the real `datetime.now(UTC)`,
    so without this the whole module goes red for ~60 hours a year — every
    test that expects an approval fails with `macro_event_pause` whenever the
    suite happens to run inside a real FOMC/CPI/NFP window. Tests that are
    ABOUT the pause monkeypatch EVENTS themselves, which overrides this.
    """
    monkeypatch.setattr(macro_calendar, "EVENTS", [])


class FakeBroker:
    """Stock venue fake: fills synchronously and remembers what was submitted."""

    name = "test"
    asset_class = "stock"

    def __init__(self, positions=(), equity="100000", cash="100000"):
        self._positions = list(positions)
        self._equity = Decimal(equity)
        self._cash = Decimal(cash)
        self.submitted: list[tuple[object, str | None]] = []
        self._by_key: dict[str, SubmittedOrder] = {}
        self._prefix = uuid.uuid4().hex[:6]  # venue ids are globally unique
        self.stops: dict[str, tuple[str, Decimal, Decimal, str]] = {}

    def submit_order(self, order, client_order_id=None):
        if client_order_id in self._by_key:
            return self._by_key[client_order_id]  # idempotent replay, like real venues
        self.submitted.append((order, client_order_id))
        oid = f"fake-{self._prefix}-{len(self.submitted)}"
        submitted = SubmittedOrder(
            broker_order_id=oid,
            status="filled",
            submitted_quantity=order.quantity,
            client_order_id=client_order_id,
            # Like Alpaca: no stop is attached to a buy; the app holds it (D-054).
        )
        if client_order_id:
            self._by_key[client_order_id] = submitted
        return submitted

    def place_stop(self, symbol, quantity, stop_price, client_order_id):
        """Record a standalone stop; a replayed key returns the same id."""
        for oid, (*_, key) in self.stops.items():
            if key == client_order_id:
                return oid
        oid = f"stop-{self._prefix}-{len(self.stops) + 1}"
        self.stops[oid] = (symbol, quantity, stop_price, client_order_id)
        return oid

    def get_order(self, broker_order_id):
        if broker_order_id in self.stops:
            return VenueOrder(status="new", terminal=False)
        n = int(broker_order_id.split("-")[2])
        order, _ = self.submitted[n - 1]
        if broker_order_id.endswith("-stop"):
            return VenueOrder(status="held", terminal=False)
        return VenueOrder(
            status="filled",
            terminal=True,
            filled_quantity=order.quantity,
            fill_price=Decimal("100"),
            filled_at=datetime.now(UTC),
        )

    def cancel_all_orders(self):
        return 0

    def get_positions(self):
        return self._positions

    def get_account(self):
        return AccountSummary(equity=self._equity, cash=self._cash)


class FakeProvider:
    """Stands in for Finnhub AND CoinGecko: canned candles + quotes.

    Candles use source='itest' — 'test' rows are EXCLUDED by
    recent_daily_candles (production-read defense), and these tests need
    their seeded candles to flow through the real read path.
    """

    def __init__(self, price="100", candle_high="101"):
        self.price = Decimal(price)
        self.candle_high = Decimal(candle_high)
        self.quote_calls = 0

    def get_quote(self, symbol):
        self.quote_calls += 1
        asset_class = "crypto" if "-USD" in symbol else "stock"
        return Quote(
            symbol=symbol,
            asset_class=asset_class,
            price=self.price,
            ts=datetime.now(UTC),
            source="test",
        )

    def get_daily_candles(self, symbol, start=None, end=None, days=None):
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return [
            Candle(
                symbol=symbol,
                asset_class="stock",
                ts=today - timedelta(days=i),
                open=Decimal("100"),
                high=self.candle_high,
                low=Decimal("99"),
                close=Decimal("100"),
                # x close 100 = $100M/day: clears the D-022 liquidity minimum.
                volume=Decimal("1000000"),
                source="itest",
            )
            for i in range(20, 0, -1)
        ]

    def close(self):
        pass


def _deps(session_factory, symbol, analyst, broker=None, redis=None, fred=None, trail=False):
    """Wire CycleDeps with fakes around the real session factory/tracker/gate.

    trail=True gives the tracker a 3% trail; the live limits have none (D-032), but the
    breach -> close machinery stays tested.
    """
    settings = Settings(_env_file=None, stock_watchlist=symbol, crypto_watchlist="")
    broker = broker or FakeBroker()
    mode = f"t-{uuid.uuid4().hex[:6]}"
    provider = FakeProvider()
    return CycleDeps(
        settings=settings,
        session_factory=session_factory,
        brokers=[broker],
        kill_switch=KillSwitch(redis or FakeRedis()),
        tracker=PortfolioTracker(
            session_factory,
            [broker],
            mode,
            limits=RiskLimits(trailing_stop_pct=Decimal("0.03")) if trail else DEFAULT_LIMITS,
        ),
        analyst=analyst,
        coingecko=provider,
        finnhub=provider,
        fred=fred,
    ), mode


def _buy(symbol):
    return TradeProposal(
        symbol=symbol, asset_class="stock", action="buy", confidence=0.9, reasoning="test"
    )


def _cycle_id():
    return f"test-{uuid.uuid4().hex[:8]}"


def _decision_row(session_factory, cycle_id, node="risk_manager"):
    with session_factory() as session:
        return (
            session.query(Decision)
            .filter(Decision.cycle_id == cycle_id, Decision.node == node)
            .one()
        )


def test_full_cycle_ingests_gates_and_executes(session_factory):
    """One cycle end to end: fake provider data in, sized+stopped ApprovedOrder
    out, submitted to the venue with a deterministic key, fill booked,
    decisions rows and snapshot persisted."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    (order,) = final["approved_orders"]
    assert order.symbol == symbol and order.side == "buy"
    assert order.quantity > 0
    assert order.stop_loss_price is not None and order.stop_loss_price < Decimal("100")
    assert final["rejections"] == []
    (result,) = final["execution_results"]
    assert result["status"] == "filled"
    assert result["client_order_id"] == f"{cycle_id}:{symbol}"
    (submitted,) = deps.brokers[0].submitted
    assert submitted[0].quantity == order.quantity
    assert final["notifications_sent"] == ["log"]
    with session_factory() as session:
        (fill,) = TradeRepository(session).fills_for_symbol(symbol, mode)
    assert fill.side == "buy" and fill.quantity == order.quantity and fill.broker == "test"

    row = _decision_row(session_factory, cycle_id, node="risk_manager")
    assert row.detail["approved"][0]["symbol"] == symbol
    # Claude's stated reason is stored with each proposal, so every trade can be explained later.
    assert row.detail["proposals"] == [
        {"symbol": symbol, "action": "buy", "confidence": 0.9, "reasoning": "test"}
    ]
    assert row.detail["equity"] == "100000"
    executor_row = _decision_row(session_factory, cycle_id, node="trade_executor")
    assert executor_row.detail["results"][0]["status"] == "filled"
    assert executor_row.detail["fills_booked"][0]["symbol"] == symbol
    # 20 ingested bars: short lookbacks present, 200-bar MAs absent.
    features = final["features"][symbol]
    assert features["close"] == 100.0
    assert "sma_10" in features and "atr_14" in features
    assert "rsi_14" not in features  # flat fake closes: RSI masked, not 100
    assert "sma_200" not in features
    with session_factory() as session:
        assert SnapshotRepository(session).snapshot_at(
            datetime.now(UTC) + timedelta(days=1), mode
        ).equity == Decimal("100000")


def test_kill_switch_active_cycle_produces_only_rejections(session_factory):
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    redis = FakeRedis()
    redis.set(KILL_SWITCH_KEY, "test halt")
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], redis=redis)
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    # Under a kill switch the analyst is not consulted (D-042): nothing to reject.
    assert final["proposals"] == [] and final["approved_orders"] == []
    assert final["execution_results"] == []
    assert deps.brokers[0].submitted == []
    assert _decision_row(session_factory, cycle_id, node="trade_executor").detail["halted"]
    # The audit row is written even on a fully halted cycle.
    assert _decision_row(session_factory, cycle_id).detail["rejected"] == []


def test_decision_row_written_even_with_no_proposals(session_factory):
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [])
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert final["approved_orders"] == [] and final["rejections"] == []
    row = _decision_row(session_factory, cycle_id)
    assert row.detail["approved"] == [] and row.detail["rejected"] == []


def test_claude_analyst_proposals_flow_through_the_gate(session_factory):
    """The ClaudeAnalyst (chain mocked — no key/network) wires into the full
    cycle: its structured output reaches the gate and comes out sized."""
    from trading_agent.graph.nodes.llm_analyst import AnalystResponse, ClaudeAnalyst

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [])

    class RisingProvider(FakeProvider):
        """60 daily closes rising 70 -> 99.5: an uptrend, so the buy survives D-038."""

        def get_daily_candles(self, symbol, start=None, end=None, days=None):
            today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            closes = [Decimal(70) + Decimal("0.5") * i for i in range(60)]
            return [
                Candle(
                    symbol=symbol,
                    asset_class="stock",
                    ts=today - timedelta(days=60 - i),
                    open=close,
                    high=close + 1,
                    low=close - 1,
                    close=close,
                    volume=Decimal("1000000"),
                    source="itest",
                )
                for i, close in enumerate(closes)
            ]

    deps.coingecko = deps.finnhub = RisingProvider()
    analyst = ClaudeAnalyst(
        Settings(_env_file=None, anthropic_api_key="test-key-not-real"),
        session_factory,
    )

    class FakeChain:
        def invoke(self, messages):
            return AnalystResponse(proposals=[_buy(symbol)])

    analyst._chain = FakeChain()
    deps.analyst = analyst
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    (order,) = final["approved_orders"]
    assert order.symbol == symbol and order.quantity > 0


def test_claude_analyst_failure_holds_the_cycle_with_two_audit_rows(session_factory):
    """LLM outage mid-cycle: the cycle completes with zero proposals, and the
    audit trail shows BOTH the analyst failure and the risk verdict."""
    from trading_agent.graph.nodes.llm_analyst import ClaudeAnalyst

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [])
    analyst = ClaudeAnalyst(
        Settings(_env_file=None, anthropic_api_key="test-key-not-real"),
        session_factory,
    )

    class DownChain:
        def invoke(self, messages):
            raise RuntimeError("anthropic api down")

    analyst._chain = DownChain()
    deps.analyst = analyst
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert final["proposals"] == [] and final["approved_orders"] == []
    with session_factory() as session:
        rows = session.query(Decision).filter(Decision.cycle_id == cycle_id).all()
    assert {r.node for r in rows} == {"llm_analyst", "risk_manager", "trade_executor"}
    (analyst_row,) = [r for r in rows if r.node == "llm_analyst"]
    assert analyst_row.detail["error"] == "RuntimeError"


def _prompted_holdings(session_factory, symbol, positions=()):
    """Run one cycle with a recording Claude chain and return the held list it was shown."""
    import json

    from trading_agent.graph.nodes.llm_analyst import AnalystResponse, ClaudeAnalyst

    deps, _ = _deps(session_factory, symbol, analyst=None, broker=FakeBroker(positions))
    analyst = ClaudeAnalyst(
        Settings(_env_file=None, anthropic_api_key="test-key-not-real"), session_factory
    )

    prompts = []

    class RecordingChain:
        def invoke(self, messages):
            """Record the snapshot sent and propose nothing."""
            prompts.append(json.loads(messages[-1].content))
            return AnalystResponse(proposals=[])

    analyst._chain = RecordingChain()
    deps.analyst = analyst
    run_cycle(build_graph(deps), cycle_id=_cycle_id())
    (prompt,) = prompts
    return prompt["held"]


def test_a_flat_account_reaches_the_analyst_as_holding_nothing(session_factory):
    """Through the full cycle, the analyst is told what the account holds."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    assert _prompted_holdings(session_factory, symbol) == []


def test_the_analyst_is_shown_its_swing_holdings_and_never_a_core_one(session_factory):
    """A held swing stock is listed; a held core stock (VOO) never reaches the prompt."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    positions = [
        BrokerPosition(symbol=s, asset_class="stock", quantity=Decimal("5"))
        for s in (symbol, "VOO")
    ]
    assert _prompted_holdings(session_factory, f"{symbol},VOO", positions) == [symbol]


def test_trailing_stop_breach_is_closed_by_the_executor(session_factory):
    """A held position whose price fell through the 3% trail surfaces as a
    stop_breach in state; the executor closes it in full and books the
    sell with realized P&L against the recorded basis."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    position = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=Decimal("5"), avg_entry_price=Decimal("100")
    )
    deps, mode = _deps(
        session_factory,
        symbol,
        analyst=lambda s: [],
        broker=FakeBroker(positions=[position]),
        trail=True,
    )
    # Fill history establishes entry time (no fills -> no trail check).
    with session_factory() as session:
        TradeRepository(session).record_fill(
            filled_at=datetime.now(UTC) - timedelta(days=10),
            symbol=symbol,
            asset_class="stock",
            side="buy",
            quantity=Decimal("5"),
            fill_price=Decimal("100"),
            broker="test",
            trading_mode=mode,
        )
        session.commit()
    # Post-entry high 200 puts the trail at 194, far above the fake price 100.
    deps.finnhub.candle_high = Decimal("200")

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    (breach,) = final["stop_breaches"]
    assert breach["symbol"] == symbol
    assert Decimal(breach["stop_price"]) == Decimal("194.00")
    (result,) = final["execution_results"]
    assert result["kind"] == "stop_breach" and result["status"] == "filled"
    ((close, key),) = deps.brokers[0].submitted
    assert close.side == "sell" and close.quantity == Decimal("5")
    assert close.stop_loss_price is None and close.source_proposal is None
    assert key.endswith(f":{symbol}:stop")
    with session_factory() as session:
        fills = TradeRepository(session).fills_for_symbol(symbol, mode)
    assert [f.side for f in fills] == ["buy", "sell"]
    assert fills[-1].realized_pnl == Decimal("0")  # fake fills at 100 = entry


def test_broker_outage_still_writes_audit_row_and_rejects_all(session_factory):
    """Portfolio assembly failing (broker down) must fail CLOSED but never
    SILENT: no approvals, every actionable proposal rejected, and the
    decisions row records the failure."""

    class DownBroker(FakeBroker):
        def get_account(self):
            raise RuntimeError("venue unreachable")

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], broker=DownBroker())
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert final["approved_orders"] == []
    (rejection,) = final["rejections"]
    assert rejection["reason"] == "evaluation_error"
    assert final["portfolio_error"] == "RuntimeError"  # watchdog input, proposal-independent
    row = _decision_row(session_factory, cycle_id)
    assert row.detail["audit"]["error"] == "portfolio_assembly:RuntimeError"
    assert row.detail["equity"] is None


def test_breached_symbol_proposals_are_preempted(session_factory):
    """A proposal for a symbol whose position breached its trail this cycle
    must never reach the executor alongside the pending close — no
    close-and-rebuy, no double-sell."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    position = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=Decimal("5"), avg_entry_price=Decimal("100")
    )
    deps, mode = _deps(
        session_factory,
        symbol,
        analyst=lambda s: [_buy(symbol)],
        broker=FakeBroker(positions=[position]),
        trail=True,
    )
    with session_factory() as session:
        TradeRepository(session).record_fill(
            filled_at=datetime.now(UTC) - timedelta(days=10),
            symbol=symbol,
            asset_class="stock",
            side="buy",
            quantity=Decimal("5"),
            fill_price=Decimal("100"),
            broker="test",
            trading_mode=mode,
        )
        session.commit()
    deps.finnhub.candle_high = Decimal("200")  # trail 194 >> price 100: breached

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    assert final["approved_orders"] == []
    (rejection,) = final["rejections"]
    assert rejection["reason"] == "stop_breach_pending"
    (breach,) = final["stop_breaches"]
    assert breach["symbol"] == symbol


def test_a_cycle_that_crashed_before_the_executor_is_replaced_not_replayed(session_factory):
    """Nothing was submitted, and its checkpoint holds prices up to hours old:
    re-invoking it runs a FRESH cycle (new id, fetch re-run) so no order is
    sized on a stale price. The crashed cycle never gets an executor row."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    boom = {"armed": True}

    def flaky_analyst(state):
        if boom.pop("armed", False):
            raise RuntimeError("transient analyst crash")
        return [_buy(symbol)]

    deps, _ = _deps(session_factory, symbol, analyst=flaky_analyst)
    cycle_id = _cycle_id()
    settings = Settings(_env_file=None)

    with postgres_checkpointer(settings.database_url) as checkpointer:
        graph = build_graph(deps, checkpointer)
        with pytest.raises(RuntimeError, match="transient analyst crash"):
            run_cycle(graph, cycle_id=cycle_id)
        quote_calls_after_crash = deps.finnhub.quote_calls

        final = run_cycle(graph, cycle_id=cycle_id)
        assert final["cycle_id"] != cycle_id
        assert deps.finnhub.quote_calls > quote_calls_after_crash  # fresh prices
        assert len(final["approved_orders"]) == 1

        rerun = run_cycle(graph, cycle_id=final["cycle_id"])  # completed: no-op replay
        assert rerun["cycle_id"] == final["cycle_id"]
        assert len(deps.brokers[0].submitted) == 1

    with session_factory() as session:
        crashed = session.query(Decision).filter(Decision.cycle_id == cycle_id).all()
        fresh = session.query(Decision).filter(Decision.cycle_id == final["cycle_id"]).all()
    assert crashed == []  # the analyst crashed before the risk node wrote anything
    assert sorted(r.node for r in fresh) == ["risk_manager", "trade_executor"]


def test_crash_inside_executor_after_venue_accept_resumes_without_double_buy(
    session_factory, monkeypatch
):
    """The executor crashes AFTER the venue accepted the order but BEFORE the
    book was written (process death, DB blip). LangGraph re-runs the whole
    node on resume: the same key must yield the same venue order, booked
    exactly once."""
    from trading_agent.graph.nodes import trade_executor

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    cycle_id = _cycle_id()
    real_book = trade_executor._book
    boom = {"armed": True}

    def flaky_book(*args, **kwargs):
        if boom.pop("armed", False):
            raise RuntimeError("db blip between venue accept and booking")
        return real_book(*args, **kwargs)

    monkeypatch.setattr(trade_executor, "_book", flaky_book)
    settings = Settings(_env_file=None)
    with postgres_checkpointer(settings.database_url) as checkpointer:
        graph = build_graph(deps, checkpointer)
        with pytest.raises(RuntimeError, match="db blip"):
            run_cycle(graph, cycle_id=cycle_id)
        final = run_cycle(graph, cycle_id=cycle_id)  # resume re-runs the executor node

    (result,) = final["execution_results"]
    assert result["status"] == "filled"
    assert len(deps.brokers[0].submitted) == 1  # venue saw ONE new order
    with session_factory() as session:
        fills = TradeRepository(session).fills_for_symbol(symbol, mode)
    assert len(fills) == 1


def test_postgres_checkpointer_roundtrips_state_models(session_factory):
    """Checkpointing must revive our pydantic state models (serde allow-list)
    with Decimal fields intact — a checkpoint that can't restore ApprovedOrder
    would break mid-cycle crash recovery once execution is real."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    cycle_id = _cycle_id()
    settings = Settings(_env_file=None)

    with postgres_checkpointer(settings.database_url) as checkpointer:
        graph = build_graph(deps, checkpointer)
        final = run_cycle(graph, cycle_id=cycle_id)
        assert len(final["approved_orders"]) == 1
        snapshot = graph.get_state({"configurable": {"thread_id": cycle_id}})

    (restored,) = snapshot.values["approved_orders"]
    assert restored.symbol == symbol
    assert isinstance(restored.quantity, Decimal) and restored.quantity > 0
    assert restored.source_proposal.action == "buy"


def test_closed_stock_market_skips_stock_quotes_so_nothing_reaches_the_gate(session_factory):
    """Scheduler says the NYSE session is closed: the fetcher takes no stock
    quotes, so a stock proposal is rejected for missing data rather than
    sized and submitted after hours."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    quote_calls_before = deps.finnhub.quote_calls

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id(), stock_market_open=False)

    assert deps.finnhub.quote_calls == quote_calls_before  # no stock quote taken
    assert final["market_data"]["prices"] == {}
    assert final["approved_orders"] == []
    (rejection,) = final["rejections"]
    assert rejection["reason"] == "insufficient_data"
    assert deps.brokers[0].submitted == []


# --- Layer-5 environment ---------------------------------------------------


class FakeFred:
    """FRED stand-in: returns a fixed close, or raises to model an outage."""

    def __init__(self, vix="17.43", days_old=1, error=None):
        self.vix = Decimal(vix)
        self.days_old = days_old
        self.error = error
        self.calls = 0

    def latest_vix(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.vix, (datetime.now(UTC) - timedelta(days=self.days_old)).date()


def _environment(session_factory, cycle_id):
    """The environment block of a cycle's risk_manager audit row."""
    return _decision_row(session_factory, cycle_id).detail["audit"]["environment"]


def test_calm_vix_is_audited_and_does_not_halve(session_factory):
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    fred = FakeFred(vix="17.43")
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], fred=fred)
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert fred.calls == 1  # one FRED call per cycle, no per-symbol fan-out
    env = _environment(session_factory, cycle_id)
    assert env["vix"] == "17.43"
    assert env["vix_date"] is not None
    assert env["macro_event_pause"] is False
    assert "vix_halved" not in _decision_row(session_factory, cycle_id).detail["audit"]
    assert len(final["approved_orders"]) == 1


def test_vix_above_threshold_halves_the_buy_and_audits_it(session_factory):
    """Layer 5's size reducer, end to end: the same proposal and the same
    portfolio, sized against a VIX above the 30 threshold, is half as large."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    calm, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], fred=FakeFred("15"))
    baseline = run_cycle(build_graph(calm), cycle_id=_cycle_id())["approved_orders"][0]

    symbol2 = f"T-{uuid.uuid4().hex[:6].upper()}"
    panic, _ = _deps(
        session_factory, symbol2, analyst=lambda s: [_buy(symbol2)], fred=FakeFred("35.2")
    )
    cycle_id = _cycle_id()
    final = run_cycle(build_graph(panic), cycle_id=cycle_id)

    (order,) = final["approved_orders"]
    assert order.quantity == baseline.quantity / 2
    detail = _decision_row(session_factory, cycle_id).detail
    assert detail["audit"]["vix_halved"] is True
    assert detail["audit"]["environment"]["vix"] == "35.2"


def test_fred_outage_leaves_vix_unknown_and_the_cycle_still_trades(session_factory):
    """VIX is a size reducer, not a safety stop: a dead feed degrades to the
    pre-existing benign default with a WARNING (never an error — an error pages
    a human, D-018) rather than halting the cycle."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    fred = FakeFred(error=ProviderError("FRED HTTP 429"))
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], fred=fred)
    cycle_id = _cycle_id()

    with capture_logs() as logs:
        final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert len(final["approved_orders"]) == 1  # cycle proceeds
    env = _environment(session_factory, cycle_id)
    assert env["vix"] is None and env["vix_date"] is None
    warning = next(log for log in logs if log["event"] == "vix_unavailable")
    assert warning["log_level"] == "warning"
    assert warning["error_type"] == "ProviderError"


def test_a_stale_vix_close_warns_but_is_still_honoured(session_factory):
    """A frozen series is warned about and KEPT. The gate has two states, so
    discarding a stale close could only ever turn halving OFF — a series that
    froze during a vol spike would resume full-size buys on day 8. Stale-calm
    already behaves exactly like unknown, so there is nothing to win the other
    way: the conservative reading is to keep halving and shout about it."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    stale = DEFAULT_LIMITS.vix_max_age_days + 3
    deps, _ = _deps(
        session_factory,
        symbol,
        analyst=lambda s: [_buy(symbol)],
        fred=FakeFred(vix="35", days_old=stale),
    )
    cycle_id = _cycle_id()

    with capture_logs() as logs:
        final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    detail = _decision_row(session_factory, cycle_id).detail
    assert detail["audit"]["environment"]["vix"] == "35"
    assert detail["audit"]["vix_halved"] is True  # still halved on a week-old panic
    assert len(final["approved_orders"]) == 1
    warning = next(log for log in logs if log["event"] == "vix_stale")
    assert warning["log_level"] == "warning" and warning["age_days"] == stale


def test_unconfigured_fred_leaves_the_environment_benign(session_factory):
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], fred=None)
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert _environment(session_factory, cycle_id)["vix"] is None
    assert len(final["approved_orders"]) == 1


def test_macro_event_pause_rejects_the_batch(session_factory, monkeypatch):
    """Inside a scheduled release window every actionable proposal is rejected
    macro_event_pause, nothing reaches the venue, and the audit says why."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    monkeypatch.setattr(
        macro_calendar, "EVENTS", [macro_calendar.MacroEvent(kind="CPI", at=datetime.now(UTC))]
    )
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], fred=FakeFred())
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert final["approved_orders"] == []
    assert [r["reason"] for r in final["rejections"]] == ["macro_event_pause"]
    assert deps.brokers[0].submitted == []
    assert _environment(session_factory, cycle_id)["macro_event_pause"] is True


def test_outside_every_window_the_macro_pause_is_off(session_factory, monkeypatch):
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    monkeypatch.setattr(
        macro_calendar,
        "EVENTS",
        [macro_calendar.MacroEvent(kind="CPI", at=datetime.now(UTC) + timedelta(days=5))],
    )
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], fred=FakeFred())
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert len(final["approved_orders"]) == 1
    assert _environment(session_factory, cycle_id)["macro_event_pause"] is False


def test_a_macro_pause_still_lets_a_stop_breach_close(session_factory, monkeypatch):
    """The pause must block NEW risk, never protection. Breaches are detected
    before the gate and go straight to the executor, so a position through its
    trail is still closed during a release window — a pause that trapped a
    losing position inside it would be the dangerous reading of Layer 5."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    monkeypatch.setattr(
        macro_calendar, "EVENTS", [macro_calendar.MacroEvent(kind="FOMC", at=datetime.now(UTC))]
    )
    position = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=Decimal("5"), avg_entry_price=Decimal("100")
    )
    deps, mode = _deps(
        session_factory,
        symbol,
        analyst=lambda s: [_buy(symbol)],
        broker=FakeBroker(positions=[position]),
        fred=FakeFred(),
        trail=True,
    )
    with session_factory() as session:
        TradeRepository(session).record_fill(
            filled_at=datetime.now(UTC) - timedelta(days=10),
            symbol=symbol,
            asset_class="stock",
            side="buy",
            quantity=Decimal("5"),
            fill_price=Decimal("100"),
            broker="test",
            trading_mode=mode,
        )
        session.commit()
    deps.finnhub.candle_high = Decimal("200")  # trail at 194, price 100

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    (breach,) = final["stop_breaches"]
    assert breach["symbol"] == symbol
    (result,) = final["execution_results"]
    assert result["kind"] == "stop_breach" and result["status"] == "filled"
    # The buy proposed the same cycle is preempted by the breach, not gated.
    assert final["approved_orders"] == []
    ((close, _),) = deps.brokers[0].submitted
    assert close.side == "sell" and close.quantity == Decimal("5")


def test_an_exhausted_calendar_warns_every_cycle(session_factory, monkeypatch):
    """The trap this guards: no upcoming events looks exactly like a calm
    market. Silence would be the bug, so the node warns instead."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    monkeypatch.setattr(
        macro_calendar,
        "EVENTS",
        [macro_calendar.MacroEvent(kind="CPI", at=datetime.now(UTC) - timedelta(days=1))],
    )
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], fred=FakeFred())

    with capture_logs() as logs:
        run_cycle(build_graph(deps), cycle_id=_cycle_id())

    (stale,) = [log for log in logs if log["event"] == "macro_calendar_stale"]
    assert stale["log_level"] == "warning"  # a warning, never a page (D-018)
    assert sorted(stale["kinds"]) == ["CPI", "FOMC", "NFP"]


def test_prune_checkpoints_drops_only_old_cycles(session_factory, engine):
    """Threads of cycles whose decisions rows predate the cutoff go; newer stay."""
    settings = Settings(_env_file=None)
    old_id, new_id = _cycle_id(), _cycle_id()
    with postgres_checkpointer(settings.database_url) as checkpointer:
        graph = build_graph(
            _deps(session_factory, f"T-{uuid.uuid4().hex[:6].upper()}", analyst=lambda s: [])[0],
            checkpointer,
        )
        run_cycle(graph, cycle_id=old_id)
        run_cycle(graph, cycle_id=new_id)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE decisions SET ts = ts - interval '40 days' WHERE cycle_id = :c"),
            {"c": old_id},
        )

    deleted = prune_checkpoints(engine, datetime.now(UTC) - timedelta(days=30))

    assert deleted > 0
    with engine.connect() as conn:
        remaining = {
            row[0]
            for row in conn.execute(
                text("SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id IN (:a, :b)"),
                {"a": old_id, "b": new_id},
            )
        }
    assert remaining == {new_id}


# --- a held position without a stop is re-protected -------------------------


def _unprotected(session_factory, redis=None):
    """A 5-share position with no stop in the book (as after /killswitch + /resume).

    Returns (deps, mode, symbol).
    """
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    position = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=Decimal("5"), avg_entry_price=Decimal("100")
    )
    deps, mode = _deps(
        session_factory,
        symbol,
        analyst=lambda s: [],
        broker=FakeBroker(positions=[position]),
        redis=redis,
    )
    return deps, mode, symbol


def test_an_unprotected_position_gets_a_stop_once(session_factory):
    """Price 100, ATR 2 (candles 99-101) -> stop 100 - 3 x 2 = 94, booked so
    the next cycle sees it and places nothing more."""
    deps, mode, symbol = _unprotected(session_factory)
    graph = build_graph(deps)

    first = run_cycle(graph, cycle_id=_cycle_id())

    (result,) = [r for r in first["execution_results"] if r["kind"] == "reprotect"]
    assert result["status"] == "placed" and result["symbol"] == symbol
    assert Decimal(result["stop_price"]) == Decimal(94)
    ((stop_symbol, qty, stop, key),) = deps.brokers[0].stops.values()
    assert (stop_symbol, qty, stop) == (symbol, Decimal("5"), Decimal(94))
    assert key.endswith(f":{symbol}:protect")
    with session_factory() as session:
        rows = OrderRepository(session).open_orders("test", mode)
    assert [(r.symbol, r.order_type, r.side) for r in rows] == [(symbol, "stop", "sell")]

    second = run_cycle(graph, cycle_id=_cycle_id())
    assert not [r for r in second["execution_results"] if r["kind"] == "reprotect"]
    assert len(deps.brokers[0].stops) == 1


def test_the_manual_kill_switch_never_reprotects(session_factory):
    """Cancel means cancel: while the human kill switch is armed, nothing is
    placed - the stop comes back on the first cycle after /resume."""
    redis = FakeRedis()
    KillSwitch(redis).activate(MANUAL_KILL_SWITCH_REASON)
    deps, _, _ = _unprotected(session_factory, redis=redis)

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    assert not [r for r in final.get("execution_results") or [] if r["kind"] == "reprotect"]
    assert deps.brokers[0].stops == {}


def _book_stop(session_factory, deps, mode, symbol, quantity, order_type="stop", side="sell"):
    """Book an open order row the fake venue (the deps' first broker) reports as working."""
    broker = deps.brokers[0]
    oid = broker.place_stop(symbol, Decimal(quantity), Decimal(90), f"pre-{uuid.uuid4().hex}")
    with session_factory() as session:
        OrderRepository(session).ensure(
            broker="test",
            broker_order_id=oid,
            created_at=datetime.now(UTC),
            symbol=symbol,
            asset_class="stock",
            side=side,
            quantity=Decimal(quantity),
            order_type=order_type,
            stop_loss_price=Decimal(90),
            status="open",
            trading_mode=mode,
        )
        session.commit()


def test_a_partly_covered_position_gets_a_stop_for_the_rest(session_factory):
    """Coverage is by quantity, summed over every stop: 2 + 2 of 5 shares under
    stops -> the venue gets a stop for the last 1."""
    deps, mode, symbol = _unprotected(session_factory)
    _book_stop(session_factory, deps, mode, symbol, "2")
    _book_stop(session_factory, deps, mode, symbol, "2")

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    (result,) = [r for r in final["execution_results"] if r["kind"] == "reprotect"]
    assert (result["status"], result["quantity"]) == ("placed", "1")
    placed = [v for v in deps.brokers[0].stops.values() if v[3].endswith(":protect")]
    assert [(sym, qty) for sym, qty, *_ in placed] == [(symbol, Decimal(1))]


@pytest.mark.parametrize(
    ("order_type", "side"), [("market", "sell"), ("stop", "buy")], ids=["market", "buy-stop"]
)
def test_only_an_open_sell_stop_counts_as_protection(session_factory, order_type, side):
    deps, mode, symbol = _unprotected(session_factory)
    _book_stop(session_factory, deps, mode, symbol, "5", order_type=order_type, side=side)

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    (result,) = [r for r in final["execution_results"] if r["kind"] == "reprotect"]
    assert (result["status"], result["quantity"]) == ("placed", "5")


@pytest.mark.parametrize(
    ("covered", "protected"), [("4.996", False), ("4.99", True)], ids=["sliver", "lot"]
)
def test_an_uncovered_sliver_below_a_tenth_of_a_percent_is_ignored(
    session_factory, covered, protected
):
    """0.004 of 5 (0.08%) is venue rounding; 0.01 (0.2%) is a naked lot."""
    deps, mode, symbol = _unprotected(session_factory)
    _book_stop(session_factory, deps, mode, symbol, covered)

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    results = [r for r in final["execution_results"] if r["kind"] == "reprotect"]
    assert bool(results) is protected


def test_a_position_sold_this_cycle_gets_no_new_stop(session_factory):
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    position = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=Decimal("5"), avg_entry_price=Decimal("100")
    )
    sell = TradeProposal(
        symbol=symbol, asset_class="stock", action="sell", confidence=0.9, reasoning="test"
    )
    deps, _ = _deps(
        session_factory, symbol, analyst=lambda s: [sell], broker=FakeBroker(positions=[position])
    )

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    assert [o.symbol for o in final["approved_orders"] if o.side == "sell"] == [symbol]
    assert not [r for r in final["execution_results"] if r["kind"] == "reprotect"]
    assert deps.brokers[0].stops == {}


def test_a_holding_outside_the_watchlist_is_deferred_quietly(session_factory):
    """No price this cycle (not watched, or stocks out of session): no stop
    from a stale close and no Discord line every 15 minutes - a log only."""
    held = BrokerPosition(
        symbol="NOTWATCHED", asset_class="stock", quantity=Decimal(5), avg_entry_price=Decimal(1)
    )
    deps, _ = _deps(
        session_factory, "T-WATCH", analyst=lambda s: [], broker=FakeBroker(positions=[held])
    )

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    assert not [r for r in final["execution_results"] if r["kind"] == "reprotect"]
    assert deps.brokers[0].stops == {}


def test_a_venue_whose_book_did_not_reconcile_is_left_alone(session_factory):
    """A stale book may hide a stop that exists: placing another would be refused
    for lack of free shares and page every cycle."""

    class FlakyBroker(FakeBroker):
        def get_order(self, broker_order_id):
            raise BrokerError("venue lookup timed out")

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    position = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=Decimal("5"), avg_entry_price=Decimal("100")
    )
    deps, mode = _deps(
        session_factory, symbol, analyst=lambda s: [], broker=FlakyBroker([position])
    )
    _book_stop(session_factory, deps, mode, symbol, "1")  # a row the lookup will fail on

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    assert "test" in final["reconcile_failed"]
    assert not [r for r in final["execution_results"] if r["kind"] == "reprotect"]


# --- the core sleeve (D-041) -----------------------------------------------------------


def _start_ledger(session_factory, mode, deposit="900"):
    """A paper ledger over FakeBroker's $100k: baseline, then a stock deposit."""
    at = datetime.now(UTC) - timedelta(minutes=5)
    with session_factory() as session:
        for kind, amount, minute in (("baseline", "-100000", 0), ("deposit", deposit, 1)):
            session.add(
                CashFlow(
                    ts=at + timedelta(minutes=minute),
                    trading_mode=mode,
                    asset_class="stock",
                    kind=kind,
                    amount=Decimal(amount),
                    # A stock-only ledger: the core's 60% of the whole deposit.
                    core_amount=Decimal(0) if kind == "baseline" else Decimal(amount) * 6 / 10,
                    equity_before=Decimal(0) if kind == "baseline" else Decimal(1),
                    scheduled_for=at.date() if kind == "deposit" else None,
                )
            )
        session.commit()


def test_the_analyst_never_trades_a_core_symbol(session_factory):
    buy_voo = TradeProposal(
        symbol="VOO", asset_class="stock", action="buy", confidence=0.9, reasoning="test"
    )
    deps, _ = _deps(session_factory, "VOO", analyst=lambda s: [buy_voo])

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    assert final["approved_orders"] == [] and final["rejections"] == []
    assert deps.brokers[0].submitted == []


def test_without_a_ledger_the_core_places_nothing(session_factory):
    deps, _ = _deps(session_factory, "VOO", analyst=lambda s: [])
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert final["approved_orders"] == []
    assert _decision_row(session_factory, cycle_id).detail["audit"]["core"] is None


def test_a_deposit_buys_the_core_booked_without_a_stop_and_never_reprotected(session_factory):
    deps, mode = _deps(session_factory, "VOO", analyst=lambda s: [])
    _start_ledger(session_factory, mode)
    graph = build_graph(deps)

    final = run_cycle(graph, cycle_id=_cycle_id())

    # 60% of the $900 deposited, x0.99 usable; VOO's 62.5% of the core stocks
    # (QQQM has no price in this test, so its share waits in cash).
    (order,) = final["approved_orders"]
    assert (order.symbol, order.side, order.stop_loss_price) == ("VOO", "buy", None)
    assert order.quantity == Decimal(540) * Decimal("0.99") * Decimal("0.625") / Decimal(100)
    (result,) = [r for r in final["execution_results"] if r["kind"] == "order"]
    assert result["status"] == "filled" and result["protective_order_id"] is None
    with session_factory() as session:
        (row,) = (
            session.query(Order).filter(Order.trading_mode == mode, Order.symbol == "VOO").all()
        )
    assert (row.side, row.order_type, row.stop_loss_price) == ("buy", "market", None)

    # Next cycle: the venue now holds VOO with no stop - the core's trail is
    # checked on closes, so re-protect leaves it alone.
    deps.brokers[0]._positions = [
        BrokerPosition(
            symbol="VOO", asset_class="stock", quantity=order.quantity, avg_entry_price=Decimal(100)
        )
    ]
    second = run_cycle(graph, cycle_id=_cycle_id())
    assert not [r for r in second["execution_results"] if r["kind"] == "reprotect"]


def test_a_core_planning_failure_never_stops_the_swing_gate(session_factory, monkeypatch):
    from trading_agent.graph.nodes import risk_manager

    def broken(*args):
        raise RuntimeError("core bug")

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    _start_ledger(session_factory, mode, deposit="90000")  # a swing sleeve that affords shares
    monkeypatch.setattr(risk_manager.core, "plan", broken)
    cycle_id = _cycle_id()

    with capture_logs() as logs:
        final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert [o.symbol for o in final["approved_orders"]] == [symbol]  # the swing buy still runs
    assert "core_plan_failed" in [e["event"] for e in logs]
    assert _decision_row(session_factory, cycle_id).detail["audit"]["core"] == {
        "error": "RuntimeError"
    }


def test_no_core_orders_are_planned_under_a_halt(session_factory):
    """The executor refuses every order under a halt; planning core buys
    anyway would post a HALTED line to Discord every cycle."""
    redis = FakeRedis()
    KillSwitch(redis).activate(MANUAL_KILL_SWITCH_REASON)
    deps, mode = _deps(session_factory, "VOO", analyst=lambda s: [], redis=redis)
    _start_ledger(session_factory, mode)
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert final["approved_orders"] == []
    assert _decision_row(session_factory, cycle_id).detail["audit"]["core"] == {
        "halted": f"kill_switch:{MANUAL_KILL_SWITCH_REASON}"
    }


def test_without_a_ledger_a_core_symbol_is_reprotected_like_any_position(
    session_factory, monkeypatch
):
    """Only the ledger hands a core holding to the close-based trail; before it,
    VOO is an ordinary position and gets its venue stop back."""
    from trading_agent.storage import repos

    # The dev database holds real Alpaca VOO bars (~$600), which would win the day
    # over this test's $100 bars and leave no stop below the $100 price.
    monkeypatch.setattr(repos, "PREFERRED_CANDLE_SOURCE", "itest")
    held = BrokerPosition(
        symbol="VOO", asset_class="stock", quantity=Decimal(5), avg_entry_price=Decimal(100)
    )
    deps, _ = _deps(session_factory, "VOO", analyst=lambda s: [], broker=FakeBroker([held]))

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    (result,) = [r for r in final["execution_results"] if r["kind"] == "reprotect"]
    assert (result["symbol"], result["status"]) == ("VOO", "placed")


def test_a_leftover_venue_stop_never_parks_a_core_symbol(session_factory):
    """Only open MARKET orders count as pending core buys/sells: a GTC stop left
    from before the ledger must not read as a sale in progress."""
    deps, mode = _deps(session_factory, "VOO", analyst=lambda s: [])
    _start_ledger(session_factory, mode)
    _book_stop(session_factory, deps, mode, "VOO", "1")

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    assert [(o.symbol, o.side) for o in final["approved_orders"]] == [("VOO", "buy")]


def test_a_core_sell_goes_out_even_when_the_venue_did_not_reconcile(session_factory):
    """A full close never depends on the book: like a breach close, the core's
    trail sell is not withheld; a buy at the same venue is."""
    from trading_agent.graph.nodes import trade_executor
    from trading_agent.graph.state import ApprovedOrder

    deps, mode = _deps(session_factory, "VOO", analyst=lambda s: [])
    sell = ApprovedOrder(
        symbol="VOO",
        asset_class="stock",
        side="sell",
        quantity=Decimal(3),
        stop_loss_price=None,
        source_proposal=None,
    )
    now = datetime.now(UTC)
    outcome, _ = trade_executor._submit(deps, "order", sell, "k-sell", {"test"}, now, mode)
    assert outcome["status"] == "filled"
    buy = sell.model_copy(update={"side": "buy"})
    outcome, _ = trade_executor._submit(deps, "order", buy, "k-buy", {"test"}, now, mode)
    assert outcome["status"] == "reconcile_failed"


def test_under_the_schedulers_crash_pause_the_core_and_its_protection_still_run(session_factory):
    """The pause stops swing orders: Claude is not asked. The core's 25% trail still
    sells (VOO, held at a cost far above any close) and an unprotected holding gets
    its stop back."""
    from trading_agent.scheduler import CRASH_PAUSE_REASON, AgentRunner

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    held = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=Decimal(5), avg_entry_price=Decimal(100)
    )
    voo = BrokerPosition(
        symbol="VOO", asset_class="stock", quantity=Decimal(1), avg_entry_price=Decimal(2000)
    )
    redis = FakeRedis()
    deps, mode = _deps(
        session_factory,
        symbol,
        analyst=lambda s: [_buy(symbol)],
        broker=FakeBroker([held, voo]),
        redis=redis,
    )
    deps.settings = deps.settings.model_copy(update={"stock_watchlist": f"VOO,{symbol}"})
    _start_ledger(session_factory, mode)
    KillSwitch(redis).start_cooldown(CRASH_PAUSE_REASON, DEFAULT_LIMITS)
    graph = build_graph(deps)
    finals = []

    def run(stocks_open, cycle_id):
        finals.append(run_cycle(graph, f"test-{cycle_id[:31]}", stock_market_open=stocks_open))
        return finals[-1]

    assert AgentRunner(deps.settings, deps, run, market_open=lambda now: True).run_once() == "ran"

    (final,) = finals
    assert final["proposals"] == []
    assert [(o.symbol, o.side) for o in final["approved_orders"]] == [("VOO", "sell")]
    by_kind = {(r["kind"], r["symbol"]): r["status"] for r in final["execution_results"]}
    assert by_kind[("order", "VOO")] == "filled"
    assert by_kind[("reprotect", symbol)] == "placed"


# --- the swing sleeve (D-042) ------------------------------------------------------------


def test_a_swing_buy_under_one_share_is_bought_fractionally_with_its_stop_stored(
    session_factory,
):
    """A $360 swing sleeve caps a position at $90: a $500 stock buys 0.18 shares, and the
    gate's ATR stop is kept on the order row, where the app holds it."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    deps.finnhub = deps.coingecko = FakeProvider(price="500")
    _start_ledger(session_factory, mode)

    final = run_cycle(build_graph(deps), cycle_id=_cycle_id())

    assert final["rejections"] == []
    (order,) = [o for o in final["approved_orders"] if o.symbol == symbol]
    assert order.quantity == Decimal("0.18")
    assert order.stop_loss_price is not None and order.stop_loss_price < Decimal(500)
    with session_factory() as session:
        row = (
            session.query(Order)
            .filter(Order.trading_mode == mode, Order.symbol == symbol, Order.side == "buy")
            .one()
        )
    assert row.stop_loss_price == order.stop_loss_price
    assert deps.brokers[0].stops == {}  # no stop at the venue: the app holds it
    held = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=order.quantity, avg_entry_price=Decimal(500)
    )
    deps.brokers[0]._positions = [held]

    # Held above its stop: re-protect leaves it to the app (no venue stop, no sale).
    quiet = run_cycle(build_graph(deps), cycle_id=_cycle_id())
    assert quiet["stop_breaches"] == []
    assert not [r for r in quiet["execution_results"] if r["kind"] == "reprotect"]
    assert deps.brokers[0].stops == {}

    # The price falls through that stop: the next cycle sells the position.
    deps.finnhub = deps.coingecko = FakeProvider(price=str(order.stop_loss_price - 1))
    second = run_cycle(build_graph(deps), cycle_id=_cycle_id())
    assert [b["symbol"] for b in second["stop_breaches"]] == [symbol]
    (close,) = [r for r in second["execution_results"] if r["kind"] == "stop_breach"]
    assert (close["symbol"], close["side"], close["status"]) == (symbol, "sell", "filled")
    assert not [r for r in second["execution_results"] if r["kind"] == "reprotect"]


def test_a_lot_closed_at_the_ledger_start_never_blends_into_a_later_sale(session_factory):
    """The cutover sells old positions without booking the sales: a lot bought and sold
    after the ledger start is measured on its own cost alone."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    deps.finnhub = deps.coingecko = FakeProvider(price="500")
    with session_factory() as session:
        TradeRepository(session).record_fill(
            filled_at=datetime.now(UTC) - timedelta(days=30),
            symbol=symbol,
            asset_class="stock",
            side="buy",
            quantity=Decimal(32),
            fill_price=Decimal(200),
            broker="test",
            trading_mode=mode,
        )
        session.commit()
    _start_ledger(session_factory, mode)
    graph = build_graph(deps)

    (order,) = [o for o in run_cycle(graph, cycle_id=_cycle_id())["approved_orders"]]
    deps.brokers[0]._positions = [
        BrokerPosition(
            symbol=symbol,
            asset_class="stock",
            quantity=order.quantity,
            avg_entry_price=Decimal(100),
        )
    ]
    deps.finnhub = deps.coingecko = FakeProvider(price=str(order.stop_loss_price - 1))
    second = run_cycle(graph, cycle_id=_cycle_id())

    (close,) = [r for r in second["execution_results"] if r["kind"] == "stop_breach"]
    assert close["status"] == "filled"
    with session_factory() as session:
        sale = TradeRepository(session).fills_for_symbol(symbol, mode)[-1]
    # FakeBroker fills every order at 100: the new lot was bought and sold at 100.
    assert (sale.side, sale.realized_pnl) == ("sell", Decimal(0))


def test_a_stop_sale_survives_a_later_failure_in_the_risk_step(session_factory, monkeypatch):
    """The stops are checked before the sleeve is measured: when measuring fails, the
    cycle still sells a swing position through its stop."""
    from trading_agent.graph.nodes import risk_manager

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    deps.finnhub = deps.coingecko = FakeProvider(price="500")
    _start_ledger(session_factory, mode)
    graph = build_graph(deps)
    (order,) = [o for o in run_cycle(graph, cycle_id=_cycle_id())["approved_orders"]]
    deps.brokers[0]._positions = [
        BrokerPosition(
            symbol=symbol,
            asset_class="stock",
            quantity=order.quantity,
            avg_entry_price=Decimal(500),
        )
    ]
    deps.finnhub = deps.coingecko = FakeProvider(price=str(order.stop_loss_price - 1))

    def broken(*args):
        raise RuntimeError("sleeve measurement failed")

    monkeypatch.setattr(risk_manager, "_swing_view", broken)
    second = run_cycle(graph, cycle_id=_cycle_id())

    assert second.get("portfolio_error") is None  # a swing failure stops swing only
    assert [b["symbol"] for b in second["stop_breaches"]] == [symbol]
    (close,) = [r for r in second["execution_results"] if r["kind"] == "stop_breach"]
    assert close["status"] == "filled"


@pytest.mark.parametrize("broken", ["_swing_view", "evaluate"])
def test_a_swing_sleeve_failure_never_stops_the_core(session_factory, monkeypatch, broken):
    """Measuring or gating the swing sleeve fails: the core still spends the deposit,
    every swing proposal is rejected, Discord is paged and the snapshot carries no
    swing figures."""
    from trading_agent.graph.nodes import risk_manager

    def fail(*args):
        raise RuntimeError("swing bug")

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    deps.settings = deps.settings.model_copy(update={"stock_watchlist": f"VOO,{symbol}"})
    _start_ledger(session_factory, mode)
    monkeypatch.setattr(risk_manager, broken, fail)
    cycle_id = _cycle_id()

    with capture_logs() as logs:
        final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert final.get("portfolio_error") is None
    assert [(o.symbol, o.side) for o in final["approved_orders"]] == [("VOO", "buy")]
    (result,) = [r for r in final["execution_results"] if r["kind"] == "order"]
    assert (result["symbol"], result["status"]) == ("VOO", "filled")
    assert [(r["symbol"], r["reason"], r["detail"]) for r in final["rejections"]] == [
        (symbol, "evaluation_error", "swing sleeve not measured: RuntimeError")
    ]
    assert ("swing_measure_failed", "error") in [(e["event"], e["log_level"]) for e in logs]
    assert _decision_row(session_factory, cycle_id).detail["audit"]["swing"] == {
        "error": "RuntimeError"
    }
    with session_factory() as session:
        (snap,) = (
            session.query(PortfolioSnapshot).filter(PortfolioSnapshot.trading_mode == mode).all()
        )
    assert snap.swing_equity is None and snap.swing_index is None


def test_a_gate_failure_without_a_ledger_fails_the_cycle_closed(session_factory, monkeypatch):
    """No ledger means no swing sleeve: the gate covers the whole account, so its
    failure fails the cycle closed and trips the scheduler's watchdog."""
    from trading_agent.graph.nodes import risk_manager

    def fail(*args):
        raise RuntimeError("gate bug")

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, _ = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    monkeypatch.setattr(risk_manager, "evaluate", fail)
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert final["portfolio_error"] == "RuntimeError"
    assert final["approved_orders"] == []
    assert [(r["symbol"], r["reason"]) for r in final["rejections"]] == [
        (symbol, "evaluation_error")
    ]
    row = _decision_row(session_factory, cycle_id)
    assert row.detail["audit"]["error"] == "portfolio_assembly:RuntimeError"


def test_with_a_ledger_the_gate_sizes_the_swing_sleeve_alone(session_factory):
    """$90,000 deposited: the core's 60% is $54,000, so the swing sleeve is
    $36,000. The buy is capped at 25% of the sleeve (90 shares at 100), not at
    10% of the account; the audit records the sleeve."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    _start_ledger(session_factory, mode, deposit="90000")
    cycle_id = _cycle_id()

    final = run_cycle(build_graph(deps), cycle_id=cycle_id)

    (order,) = [o for o in final["approved_orders"] if o.symbol == symbol]
    assert order.quantity == Decimal(90)
    swing_audit = _decision_row(session_factory, cycle_id).detail["audit"]["swing"]
    assert Decimal(swing_audit["equity"]) == Decimal(36000)
    assert Decimal(swing_audit["index"]) == Decimal(1)
    with session_factory() as session:
        snap = SnapshotRepository(session).latest(mode)
    assert (snap.swing_equity, snap.swing_index) == (Decimal(36000), Decimal(1))


def test_a_swing_loss_trips_the_swing_breaker_while_the_core_keeps_buying(session_factory):
    """The sleeve's index fell ~5% since the day started: the daily breaker
    stops swing entries (and arms a cooldown), but the core's deposit buy
    still goes out under that cooldown (D-040 item 6)."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    redis = FakeRedis()
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)], redis=redis)
    _start_ledger(session_factory, mode, deposit="90000")
    graph = build_graph(deps)
    run_cycle(graph, cycle_id=_cycle_id())  # index 1 at the sleeve's 36,000 (after the buy: less)
    with session_factory() as session:  # the day's first snapshot: the sleeve was 5% higher
        first = SnapshotRepository(session).latest(mode)
        session.add(
            PortfolioSnapshot(
                ts=datetime.now(UTC) - timedelta(minutes=4),  # after the ledger's start
                equity=first.equity,
                cash=first.cash,
                positions={},
                peak_equity=first.peak_equity,
                trading_mode=mode,
                swing_equity=first.swing_equity,
                swing_index=Decimal("1.05"),
            )
        )
        session.add(  # a new deposit for the core to spend
            CashFlow(
                ts=datetime.now(UTC),
                trading_mode=mode,
                asset_class="stock",
                kind="deposit",
                amount=Decimal(1000),
                core_amount=Decimal(600),
                equity_before=first.equity,
                scheduled_for=(datetime.now(UTC) + timedelta(days=14)).date(),
            )
        )
        session.commit()
    deps.analyst = lambda s: [_buy(f"T-{uuid.uuid4().hex[:6].upper()}")]
    # VOO priced from now on (the fake venue's cash never moves, so no core fill in cycle 1).
    deps.settings = deps.settings.model_copy(update={"stock_watchlist": f"VOO,{symbol}"})
    cycle_id = _cycle_id()

    final = run_cycle(graph, cycle_id=cycle_id)

    assert [r["reason"] for r in final["rejections"]] == ["daily_loss_breaker"]
    assert COOLDOWN_KEY in redis.store
    core_results = [r for r in final["execution_results"] if r["symbol"] == "VOO"]
    assert core_results and all(r["status"] == "filled" for r in core_results)


def _swing_fixture(session_factory, mode):
    """A ledger started Monday 2026-09-21 10:00 UTC with swing snapshots after it.

    Monday 12:00 index 1.2 (the week's anchor), Tuesday 12:00 index 1.3 (the
    peak), Tuesday 23:00 index 1.1 with swing equity 360 (the day's anchor and
    the last measurement), then a $500 deposit on Wednesday 14:00. Returns (the
    ledger rows, Wednesday 15:00).
    """
    from trading_agent.storage.repos import CashFlowRepository

    start = datetime(2026, 9, 21, 10, tzinfo=UTC)
    wednesday = start + timedelta(days=2, hours=5)
    rows = [
        ("baseline", "stock", "-100000", "0", start, None),
        ("deposit", "stock", "900", "540", start + timedelta(minutes=1), start.date()),
        ("deposit", "crypto", "50", "50", wednesday - timedelta(hours=1), wednesday.date()),
        ("deposit", "stock", "450", "300", wednesday - timedelta(hours=1), wednesday.date()),
    ]
    with session_factory() as session:
        for kind, venue, amount, core_part, ts, day in rows:
            session.add(
                CashFlow(
                    ts=ts,
                    trading_mode=mode,
                    asset_class=venue,
                    kind=kind,
                    amount=Decimal(amount),
                    core_amount=Decimal(core_part),
                    equity_before=Decimal(1),
                    scheduled_for=day,
                )
            )
        snaps = SnapshotRepository(session)
        for hours, index in ((2, "1.2"), (26, "1.3"), (37, "1.1")):
            snaps.record(
                ts=start + timedelta(hours=hours, microseconds=uuid.uuid4().int % 999_999),
                equity=Decimal(1000),
                cash=Decimal(1000),
                positions={},
                peak_equity=Decimal(1000),
                trading_mode=mode,
                swing=(Decimal(360), Decimal(index)),
            )
        trades = TradeRepository(session)
        for at, symbol, pnl in (
            (start - timedelta(hours=1), "T-OLD", "-1"),  # before the ledger: not counted
            (wednesday - timedelta(hours=5), "T-SWING", "-2"),
            (wednesday - timedelta(hours=4), "VOO", "-3"),  # a core close: not counted
            (wednesday - timedelta(hours=3), "T-SWING", "-1"),
        ):
            trades.record_fill(
                filled_at=at,
                symbol=symbol,
                asset_class="stock",
                side="sell",
                quantity=Decimal(1),
                fill_price=Decimal(100),
                broker="test",
                trading_mode=mode,
                realized_pnl=Decimal(pnl),
            )
        session.commit()
        flows = CashFlowRepository(session).ledger(mode)
    return flows, wednesday


def _measure(session_factory, mode, flows, now, stale=False, open_orders=(), stock_cash="1350"):
    """Run the risk node's swing measurement on a 1,350 stock venue and write its snapshot.

    Returns (the gate's view, the audit dict).
    """
    from trading_agent.graph.nodes import risk_manager
    from trading_agent.portfolio import core
    from trading_agent.risk.models import PortfolioState

    deps = SimpleNamespace(
        tracker=PortfolioTracker(session_factory, [], mode), session_factory=session_factory
    )
    account = PortfolioState(
        equity=Decimal(1400),
        cash=Decimal(1350),
        peak_equity=Decimal(1400),
        day_start_equity=Decimal(1400),
        week_start_equity=Decimal(1400),
        consecutive_losses=0,
        venue_cash={"stock": Decimal(stock_cash), "crypto": Decimal(50)},
    )
    book = risk_manager._CoreBook(
        flows=flows,
        candles={},
        fills={s: [] for s in core.CORE_TARGETS},
        open_orders=list(open_orders),
        stock_fills=[],
    )
    marks = {"VOO": Decimal(100)}  # an open VOO buy has a price: it must not leave the claim
    view, audit, record = risk_manager._swing_view(deps, account, marks, book, now, stale)
    deps.tracker.snapshot(account, now, swing=record)
    return view, audit


def test_the_swing_view_measures_the_index_and_anchors_from_the_book(session_factory):
    """Deposit neutral (the stock rows less the core's 60%: 450 - 300 = 150 new);
    day/week/peak anchors each from their own snapshot; swing closes only, since
    the ledger started; open core buys stay in the core's claim."""
    mode = f"t-{uuid.uuid4().hex[:6]}"
    flows, now = _swing_fixture(session_factory, mode)
    pending = Order(
        created_at=now,
        symbol="VOO",
        asset_class="stock",
        side="buy",
        quantity=Decimal(1),
        order_type="market",
        broker="test",
    )

    view, audit = _measure(session_factory, mode, flows, now, open_orders=[pending])

    # 1,350 at the venue - 60% of 1,400 deposited = 510 = 360 + the 150 new: no return.
    assert (view.equity, view.cash, Decimal(audit["index"])) == (
        Decimal(510),
        Decimal(510),
        Decimal("1.1"),
    )
    assert view.day_start_equity == Decimal(510)
    assert view.week_start_equity == Decimal(510) * Decimal("1.2") / Decimal("1.1")
    assert view.peak_equity == Decimal(510) * Decimal("1.3") / Decimal("1.1")
    assert view.consecutive_losses == 2
    assert view.risk_equity == Decimal(1400)
    assert Decimal(audit["unexplained"]) == 0  # the 1,350 stock deposited, nothing spent
    with session_factory() as session:
        latest = SnapshotRepository(session).latest(mode)
    assert (latest.ts, latest.swing_index) == (now, Decimal("1.1"))


def test_a_stale_book_does_not_measure_the_sleeve(session_factory):
    """An unreconciled venue could hide a fill that would read as a return: the
    last measurement stands and this cycle's snapshot records no swing figures."""
    mode = f"t-{uuid.uuid4().hex[:6]}"
    flows, now = _swing_fixture(session_factory, mode)

    view, audit = _measure(session_factory, mode, flows, now, stale=True)

    assert (view.equity, Decimal(audit["index"])) == (Decimal(360), Decimal("1.1"))
    assert view.cash == Decimal(0)  # no buying on numbers that may hold the core's money
    assert audit["measured"] == "False"
    with session_factory() as session:
        snaps = SnapshotRepository(session)
        assert snaps.latest(mode).swing_index is None
        assert snaps.latest_swing(mode, flows[0].ts).ts < now


def test_with_a_ledger_the_book_is_reconciled_before_the_sleeves_are_measured(
    session_factory, monkeypatch
):
    """Last cycle's core fills must be on the book before the swing sleeve is
    measured, or they would read as swing gains/losses; a failed venue lookup
    leaves the sleeve unmeasured."""
    from trading_agent.graph.nodes import risk_manager

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [])
    _start_ledger(session_factory, mode, deposit="90000")
    calls = []
    real_build = deps.tracker.build_state
    monkeypatch.setattr(
        risk_manager,
        "reconcile_open_orders",
        lambda d, m: calls.append("reconcile") or ([], {"test"}),
    )
    monkeypatch.setattr(
        deps.tracker, "build_state", lambda *a, **k: calls.append("build") or real_build(*a, **k)
    )
    cycle_id = _cycle_id()

    run_cycle(build_graph(deps), cycle_id=cycle_id)

    assert calls[:2] == ["reconcile", "build"]
    swing_audit = _decision_row(session_factory, cycle_id).detail["audit"]["swing"]
    assert swing_audit["measured"] == "True"  # nothing measured before: the first reading stands


def test_an_unmeasured_snapshot_at_the_boundary_never_blinds_the_day_breaker(session_factory):
    """A cycle that could not measure the sleeve just before midnight must not
    become the day's anchor: the last MEASURED one (index 1.1) is."""
    mode = f"t-{uuid.uuid4().hex[:6]}"
    flows, now = _swing_fixture(session_factory, mode)
    with session_factory() as session:
        SnapshotRepository(session).record(
            ts=now.replace(hour=0) - timedelta(minutes=1),
            equity=Decimal(1000),
            cash=Decimal(1000),
            positions={},
            peak_equity=Decimal(1000),
            trading_mode=mode,
        )
        session.commit()

    # 474 - the 150 deposited = 324 = the 360 measured last, less 10%: a real loss today.
    view, _ = _measure(session_factory, mode, flows, now, stock_cash="1314")

    assert round(1 - view.equity / view.day_start_equity, 6) == Decimal("0.1")


def test_off_hours_a_swing_position_is_valued_at_its_close_not_its_cost(session_factory):
    """No stock prices outside the session: the sleeve marks a held position at
    its newest bar (refreshed by this cycle's ingest), never at its cost."""
    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    held = BrokerPosition(
        symbol=symbol, asset_class="stock", quantity=Decimal(3), avg_entry_price=Decimal(50)
    )
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [], broker=FakeBroker([held]))
    _start_ledger(session_factory, mode, deposit="90000")
    cycle_id = _cycle_id()

    run_cycle(build_graph(deps), cycle_id=cycle_id, stock_market_open=False)

    audit = _decision_row(session_factory, cycle_id).detail["audit"]["swing"]
    # 36,000 swing cash + 3 x the stored close of 100 (not 3 x the 50 cost).
    assert Decimal(audit["equity"]) == Decimal(36300)
    assert audit["stale"] == []


def test_a_failed_stock_venue_lookup_leaves_the_sleeve_unmeasured_and_unfunded(
    session_factory, monkeypatch
):
    from trading_agent.graph.nodes import risk_manager

    symbol = f"T-{uuid.uuid4().hex[:6].upper()}"
    deps, mode = _deps(session_factory, symbol, analyst=lambda s: [_buy(symbol)])
    _start_ledger(session_factory, mode, deposit="90000")
    graph = build_graph(deps)
    run_cycle(graph, cycle_id=_cycle_id())  # a first measurement
    monkeypatch.setattr(risk_manager, "reconcile_open_orders", lambda d, m: ([], {"test"}))
    cycle_id = _cycle_id()

    final = run_cycle(graph, cycle_id=cycle_id)

    audit = _decision_row(session_factory, cycle_id).detail["audit"]["swing"]
    assert (audit["measured"], audit["stale"]) == ("False", ["test"])
    assert not [o for o in final["approved_orders"] if o.symbol == symbol]


def test_the_sleeves_new_money_is_the_swing_part_each_deposit_recorded(
    session_factory, monkeypatch
):
    """A later change to the core's share moves nothing already deposited: the
    index still reads the 150 recorded as swing's, so no return."""
    from trading_agent.portfolio import ledger

    monkeypatch.setattr(ledger, "CORE_STOCK_SHARE", Decimal("0.5"))
    mode = f"t-{uuid.uuid4().hex[:6]}"
    flows, now = _swing_fixture(session_factory, mode)

    view, audit = _measure(session_factory, mode, flows, now)

    assert (view.equity, Decimal(audit["index"])) == (Decimal(510), Decimal("1.1"))
