"""Hermetic graph tests: topology and stub-node contracts (no infrastructure).

The full-cycle path (risk node needs TimescaleDB) lives in
test_graph_integration.py.
"""

from decimal import Decimal
from itertools import pairwise

from fakes import FakeRedis
from trading_agent.graph.builder import NODE_ORDER, build_graph
from trading_agent.graph.deps import CycleDeps
from trading_agent.graph.nodes.feature_engine import compute_features
from trading_agent.graph.nodes.llm_analyst import analyze, stub_analyst
from trading_agent.graph.nodes.ml_predictor import predict
from trading_agent.graph.state import TradeProposal
from trading_agent.risk.killswitch import KILL_SWITCH_KEY, KillSwitch


def _bare_deps(analyst=stub_analyst) -> CycleDeps:
    """CycleDeps with no live resources — enough to compile the graph and run
    nodes that never touch infrastructure."""
    return CycleDeps(
        settings=None,
        session_factory=None,
        brokers=[],
        kill_switch=KillSwitch(FakeRedis()),
        tracker=None,
        analyst=analyst,
        coingecko=None,
        finnhub=None,
    )


def test_graph_compiles_with_full_cycle_topology():
    graph = build_graph(_bare_deps())
    nodes = set(graph.get_graph().nodes)
    assert set(NODE_ORDER) <= nodes
    # Straight-line shape: every node except the last feeds exactly the next.
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    for prev, nxt in pairwise(NODE_ORDER):
        assert (prev, nxt) in edges


def test_feature_and_predictor_nodes_degrade_to_empty_maps(tmp_path):
    deps = _bare_deps()
    deps.model_dir = tmp_path  # no model trained
    assert compute_features({"cycle_id": "c"}, deps=deps) == {"features": {}}
    assert predict({"cycle_id": "c"}, deps=deps) == {"predictions": {}}


def test_stub_analyst_never_proposes_trades():
    """The stub must be incapable of trading: holds only, zero confidence.
    A stub that emitted buy/sell would be a live money path with no
    intelligence behind it."""
    state = {
        "market_data": {
            "prices": {"BRK-B": Decimal("100"), "BTC-USD": Decimal("60000")},
            # Asset classes come from quotes, not symbol shape: BRK-B is a hyphenated STOCK.
            "asset_classes": {"BRK-B": "stock", "BTC-USD": "crypto"},
        },
    }
    proposals = stub_analyst(state)
    assert [p.symbol for p in proposals] == ["BRK-B", "BTC-USD"]
    assert all(p.action == "hold" and p.confidence == 0.0 for p in proposals)
    assert {p.symbol: p.asset_class for p in proposals} == {
        "BRK-B": "stock",
        "BTC-USD": "crypto",
    }


def test_stub_analyst_with_no_prices_proposes_nothing():
    assert stub_analyst({}) == []


def test_analyze_node_returns_configured_analyst_output():
    proposal = TradeProposal(
        symbol="AAPL", asset_class="stock", action="buy", confidence=0.9, reasoning="test"
    )
    deps = _bare_deps(analyst=lambda state: [proposal])
    assert analyze({"cycle_id": "c"}, deps=deps) == {"proposals": [proposal]}


def test_the_analyst_is_not_consulted_under_a_kill_switch():
    """Under the automatic kill switch the gate would reject everything, so the cycle
    runs the core and the stops (D-042) without LLM spend."""

    def never(state):
        raise AssertionError("must not be called")

    deps = _bare_deps(analyst=never)
    deps.kill_switch.redis.set(KILL_SWITCH_KEY, "max_drawdown_exceeded")
    assert analyze({"cycle_id": "c"}, deps=deps) == {"proposals": []}
