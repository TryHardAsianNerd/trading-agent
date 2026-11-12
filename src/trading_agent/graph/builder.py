"""Wires the nodes into a straight-line LangGraph cycle, one checkpointer thread per cycle.

Every cycle runs every node, so a no-trade cycle still snapshots, checks stops and
writes its decisions row. A cycle that crashed in the executor resumes under its
original idempotency keys; one that crashed earlier is replaced, never replayed (D-037).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import partial
from itertools import pairwise
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from trading_agent.graph.deps import CycleDeps
from trading_agent.graph.nodes.data_fetcher import fetch
from trading_agent.graph.nodes.feature_engine import compute_features
from trading_agent.graph.nodes.llm_analyst import analyze
from trading_agent.graph.nodes.ml_predictor import predict
from trading_agent.graph.nodes.notifier import notify
from trading_agent.graph.nodes.risk_manager import apply_risk_gate
from trading_agent.graph.nodes.trade_executor import execute
from trading_agent.graph.state import AgentState
from trading_agent.log import get_logger

log = get_logger(__name__)

NODE_ORDER = ["fetch", "features", "predict", "analyze", "risk", "execute", "notify"]
# Nodes a crashed cycle is resumed at: from here on, venue submissions may exist.
RESUME_FROM = frozenset(NODE_ORDER[NODE_ORDER.index("execute") :])

_NODE_FNS = {
    "fetch": fetch,
    "features": compute_features,
    "predict": predict,
    "analyze": analyze,
    "risk": apply_risk_gate,
    "execute": execute,
    "notify": notify,
}

# Checkpointed-state models; the serializer must never revive types outside this allow-list.
_CHECKPOINT_TYPES = [
    ("trading_agent.graph.state", "TradeProposal"),
    ("trading_agent.graph.state", "ApprovedOrder"),
]


def build_graph(
    deps: CycleDeps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Compile the cycle graph with deps bound into every node (no checkpointer in tests)."""
    graph = StateGraph(AgentState)
    for name in NODE_ORDER:
        graph.add_node(name, partial(_NODE_FNS[name], deps=deps))
    graph.add_edge(START, NODE_ORDER[0])
    for prev, nxt in pairwise(NODE_ORDER):
        graph.add_edge(prev, nxt)
    graph.add_edge(NODE_ORDER[-1], END)
    return graph.compile(checkpointer=checkpointer)


def run_cycle(
    graph: CompiledStateGraph, cycle_id: str | None = None, *, stock_market_open: bool = True
) -> AgentState:
    """Run one cycle and return its final state; idempotent per cycle_id.

    A cycle_id that crashed in or after the executor resumes from its checkpoint
    (LangGraph resumes only on a None input). One that crashed earlier holds prices
    that may be hours old, so a fresh cycle runs under a new id instead. A completed
    cycle_id returns its stored state. stock_market_open is ignored on a resume.
    """
    cycle_id = cycle_id or str(uuid4())
    config = {"configurable": {"thread_id": cycle_id}}
    if graph.checkpointer is not None:
        snapshot = graph.get_state(config)
        if snapshot.next and RESUME_FROM.issuperset(snapshot.next):
            log.warning("cycle_resumed_after_crash", cycle_id=cycle_id, next=snapshot.next)
            return graph.invoke(None, config=config)
        if snapshot.next:
            fresh = str(uuid4())
            log.warning(
                "cycle_replaced_after_crash",
                cycle_id=cycle_id,
                next=snapshot.next,
                new_cycle_id=fresh,
                note="nothing was submitted; not replayed on stale prices",
            )
            cycle_id, config = fresh, {"configurable": {"thread_id": fresh}}
        elif snapshot.values:
            log.warning("cycle_already_complete", cycle_id=cycle_id)
            return snapshot.values
    initial: AgentState = {
        "cycle_id": cycle_id,
        "cycle_started_at": datetime.now(UTC),
        "stock_market_open": stock_market_open,
    }
    return graph.invoke(initial, config=config)


@contextmanager
def postgres_checkpointer(database_url: str) -> Iterator[PostgresSaver]:
    """Yield a set-up PostgresSaver that revives only the allow-listed state models.

    Its pool health-checks connections on checkout, so a long-running process
    survives a database restart.
    """
    # psycopg wants plain postgresql://, without SQLAlchemy's "+driver" marker.
    scheme, _, rest = database_url.partition("://")
    conn_str = f"{scheme.partition('+')[0]}://{rest}"
    # autocommit + dict_row are PostgresSaver requirements (setup() runs DDL).
    pool = ConnectionPool(
        conn_str,
        min_size=1,
        max_size=2,
        kwargs={"autocommit": True, "row_factory": dict_row},
        check=ConnectionPool.check_connection,
        open=False,
    )
    with pool:
        saver = PostgresSaver(
            pool, serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)
        )
        saver.setup()
        yield saver
