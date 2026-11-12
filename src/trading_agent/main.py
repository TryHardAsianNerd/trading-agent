"""Entry point: python -m trading_agent.main [--once]

By default runs the agent (15-minute cycles, the Discord command bot, daily checkpoint
pruning) until Ctrl+C; --once runs a single cycle and exits.
"""

import argparse
import asyncio

from trading_agent import __version__
from trading_agent.config import load_settings
from trading_agent.graph.builder import build_graph, postgres_checkpointer, run_cycle
from trading_agent.graph.deps import make_deps
from trading_agent.log import configure_logging, drain_alerts, get_logger
from trading_agent.notify.discord import format_crash_alert
from trading_agent.scheduler import serve, stocks_open
from trading_agent.storage.db import init_db, make_engine, make_session_factory


def main(argv: list[str] | None = None) -> None:
    """Configure logging, wire dependencies, then serve until Ctrl+C or run one cycle."""
    parser = argparse.ArgumentParser(prog="trading_agent")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = parser.parse_args(argv)

    configure_logging()
    log = get_logger("trading_agent.main")

    settings = load_settings()
    log.info(
        "startup",
        version=__version__,
        trading_mode=settings.trading_mode,
        stock_watchlist=settings.stock_symbols,
        crypto_watchlist=settings.crypto_symbols,
        cycle_interval_minutes=settings.cycle_interval_minutes,
        once=args.once,
    )

    if not settings.is_paper:
        # Unreachable in this paper-only build: config rejects live mode.
        log.warning("live_mode_active", note="live trading mode — Phase 4 only")

    engine = make_engine(settings.database_url)
    init_db(engine)
    deps = make_deps(settings, make_session_factory(engine))

    with postgres_checkpointer(settings.database_url) as checkpointer:
        graph = build_graph(deps, checkpointer)
        if not args.once:
            try:
                asyncio.run(serve(settings, deps, graph, engine))
            except KeyboardInterrupt:
                log.info("shutdown_requested")
            return
        try:
            final = run_cycle(graph, stock_market_open=stocks_open(settings))
        except Exception as e:
            # The notifier node never ran: page with the errors buffered so far.
            if deps.notifier is not None:
                deps.notifier.send(format_crash_alert(e, drain_alerts(), settings.trading_mode))
            raise

    log.info(
        "cycle_complete",
        cycle_id=final.get("cycle_id"),
        approved=len(final.get("approved_orders") or []),
        rejected=len(final.get("rejections") or []),
        executed=len(final.get("execution_results") or []),
    )


if __name__ == "__main__":
    main()
