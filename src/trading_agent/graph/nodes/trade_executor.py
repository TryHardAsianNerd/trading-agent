"""Node 6, trade executor: the only place orders reach a venue. Per cycle, in order:

1. Kill-switch check of its own; the manual kill switch cancels every open order first.
2. Reconcile: look up every open orders row at its venue and book any fill.
3. Re-protect: give any held quantity no open stop covers a venue stop at price - N x ATR.
4. Submit stop-breach sales, then approved orders, keyed by cycle_id and symbol.
5. Book each submission's orders rows, then reconcile them at once.
6. Write one decisions row, whatever happened.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from trading_agent.brokers.base import (
    Broker,
    BrokerError,
    BrokerPosition,
    SubmittedOrder,
    UnfillableQuantityError,
    UnknownOrderError,
    UnprotectedPositionError,
    WrongVenueError,
)
from trading_agent.graph.state import AgentState, ApprovedOrder
from trading_agent.log import get_logger
from trading_agent.portfolio import core
from trading_agent.portfolio.tracker import average_cost_walk
from trading_agent.risk.killswitch import MANUAL_KILL_SWITCH_REASON, stops_everything
from trading_agent.risk.limits import DEFAULT_LIMITS
from trading_agent.risk.stops import atr_stop_price, wilder_atr
from trading_agent.storage.models import Order, Trade
from trading_agent.storage.repos import (
    CashFlowRepository,
    DecisionRepository,
    OhlcvRepository,
    OrderRepository,
    TradeRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from trading_agent.graph.deps import CycleDeps

log = get_logger(__name__)

# An instant retry mostly repeats the same transient failure.
RETRY_DELAY_SECONDS = 1.0


def execute(state: AgentState, *, deps: CycleDeps) -> dict:
    """Run the executor's steps; return the re-protect and submission results and reconcile_failed.

    reconcile_failed lists the venues whose books didn't reconcile; the watchdog reads it.
    """
    cycle_id = state["cycle_id"]
    now = datetime.now(UTC)
    mode = deps.tracker.trading_mode

    halted, halt_reason = deps.kill_switch.is_halted()
    # Cancel before reconciling, so this cycle's poll retires the cancelled rows and a DB
    # outage cannot keep the kill switch from the venues. Only the manual kill switch
    # cancels (D-020): every other halt must leave the protective stops working.
    manual_halt = halt_reason == f"kill_switch:{MANUAL_KILL_SWITCH_REASON}"
    cancelled = _cancel_everywhere(deps) if manual_halt else None
    booked, failed_brokers = reconcile_open_orders(deps, mode)
    results: list[dict[str, object]] = []
    protective = not stops_everything(halted, halt_reason)

    submissions: list[tuple[str, ApprovedOrder, str]] = [
        (
            "stop_breach",
            ApprovedOrder(
                symbol=b["symbol"],
                asset_class=b["asset_class"],
                side="sell",
                quantity=Decimal(str(b["quantity"])),
                stop_loss_price=None,
                source_proposal=None,
            ),
            f"{cycle_id}:{b['symbol']}:stop",
        )
        for b in state.get("stop_breaches") or []
    ] + [("order", o, f"{cycle_id}:{o.symbol}") for o in state.get("approved_orders") or []]
    if protective:
        closing = {
            o.symbol
            for kind, o, _ in submissions
            if o.side == "sell" and _goes_out(kind, o, halted, protective)
        }
        results += _reprotect(deps, state, closing, failed_brokers, now, mode)

    for kind, order, client_order_id in submissions:
        result: dict[str, object] = {
            "kind": kind,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": str(order.quantity),
            "client_order_id": client_order_id,
        }
        if not _goes_out(kind, order, halted, protective):
            result.update(status="halted", detail=halt_reason)
        else:
            outcome, fills = _submit(deps, kind, order, client_order_id, failed_brokers, now, mode)
            result.update(outcome)
            booked += fills
        log.info("execution_result", cycle_id=cycle_id, **result)
        results.append(result)

    if halted:
        log.warning("execution_halted", cycle_id=cycle_id, reason=halt_reason)

    with deps.session_factory() as session:
        DecisionRepository(session).log(
            cycle_id=cycle_id,
            ts=now,
            node="trade_executor",
            detail={
                "halted": halted,
                "halt_reason": halt_reason if halted else None,
                "cancelled": cancelled,
                "results": results,
                "fills_booked": booked,
                "reconcile_failed": sorted(failed_brokers),
            },
        )
        session.commit()
    return {"execution_results": results, "reconcile_failed": sorted(failed_brokers)}


def _goes_out(kind: str, order: ApprovedOrder, halted: bool, protective: bool) -> bool:
    """Whether a submission goes out under the current halt state.

    Stop-breach sales and the core's orders go out when protective (the halt does not
    stop everything); swing entries and signal sells stop under every halt.
    """
    if not halted:
        return True
    return protective and (kind == "stop_breach" or core.is_core(order.symbol))


def _reprotect(
    deps: CycleDeps,
    state: AgentState,
    closing: set[str],
    failed_brokers: set[str],
    now: datetime,
    mode: str,
) -> list[dict[str, object]]:
    """Give every held quantity that no open stop in the book covers a venue stop.

    Coverage is summed by quantity, so an added lot whose stop failed is covered too.
    Nothing else replaces a stop the manual kill switch cancelled (D-036). Skipped:
    unreconciled venues, positions closing this cycle, positions whose stop the app holds,
    and unpriced ones (a stop sized from a stale close can land above the market).
    """
    prices = (state.get("market_data") or {}).get("prices") or {}
    results: list[dict[str, object]] = []
    for broker in deps.brokers:
        if broker.name in failed_brokers:
            continue
        try:
            positions = broker.get_positions()
            with deps.session_factory() as session:
                # With a ledger the core's stop is its close-based trail (portfolio.core);
                # without one a core symbol is an ordinary position and needs a venue stop.
                core_managed = bool(CashFlowRepository(session).ledger(mode))
                book = OrderRepository(session)
                covered: dict[str, Decimal] = {}
                for o in book.open_orders(broker.name, mode):
                    if o.order_type == "stop" and o.side == "sell":
                        covered[o.symbol] = covered.get(o.symbol, Decimal(0)) + o.quantity
                app_held = {
                    p.symbol
                    for p in positions
                    if not core.is_core(p.symbol) and book.entry_stop(p.symbol, mode) is not None
                }
        except Exception as e:
            log.warning("reprotect_skipped", broker=broker.name, reason=type(e).__name__)
            continue
        for position in positions:
            uncovered = position.quantity - covered.get(position.symbol, Decimal(0))
            # A sliver below venue lot rounding is not a naked position.
            if (
                position.symbol in closing
                or position.symbol in app_held
                or (core_managed and core.is_core(position.symbol))
                or uncovered <= position.quantity * Decimal("0.001")
            ):
                continue
            price = prices.get(position.symbol)
            if price is None:
                log.warning("reprotect_deferred", symbol=position.symbol, reason="no_price")
                continue
            results.append(_protect(deps, broker, position, uncovered, price, state, now, mode))
    return results


def _protect(
    deps: CycleDeps,
    broker: Broker,
    position: BrokerPosition,
    quantity: Decimal,
    price: Decimal,
    state: AgentState,
    now: datetime,
    mode: str,
) -> dict[str, object]:
    """Place and book one protective stop for quantity and return its result; never raises."""
    symbol = position.symbol
    key = f"{state['cycle_id']}:{symbol}:protect"
    result: dict[str, object] = {
        "kind": "reprotect",
        "symbol": symbol,
        "side": "sell",
        "quantity": format(quantity.normalize(), "f"),
        "client_order_id": key,
    }
    try:
        with deps.session_factory() as session:
            candles = OhlcvRepository(session).recent_daily_candles(symbol, limit=60)
        stop = atr_stop_price(price, wilder_atr(candles), DEFAULT_LIMITS)
        if stop <= 0:
            raise ValueError("ATR stop not positive")
    except Exception as e:
        log.error(
            "position_unprotected",
            symbol=symbol,
            reason="no_stop_computable",
            error_type=type(e).__name__,
        )
        result.update(status="unprotected", detail=f"no stop computable ({type(e).__name__})")
        return result
    try:
        stop_id = broker.place_stop(symbol, quantity, stop, key)
    except UnfillableQuantityError:
        log.warning("reprotect_unfillable", symbol=symbol, quantity=str(quantity))
        result.update(status="unprotected", detail="below the venue's minimum size")
        return result
    except Exception as e:
        log.error("position_unprotected", symbol=symbol, broker=broker.name, error=str(e))
        result.update(status="unprotected", detail=str(e))
        return result
    result.update(stop_price=str(stop), broker_order_id=stop_id)
    try:
        with deps.session_factory() as session:
            OrderRepository(session).ensure(
                broker=broker.name,
                broker_order_id=stop_id,
                created_at=now,
                symbol=symbol,
                asset_class=position.asset_class,
                side="sell",
                quantity=quantity,
                order_type="stop",
                stop_loss_price=stop,
                status="open",
                trading_mode=mode,
            )
            session.commit()
    except Exception:
        # The stop IS at the venue; only our book misses it (same policy as _book).
        log.exception("submission_unbooked", symbol=symbol, broker_order_id=stop_id)
        result.update(status="placed_unbooked")
        return result
    log.warning("position_reprotected", symbol=symbol, broker=broker.name, stop=str(stop))
    result.update(status="placed")
    return result


def _cancel_everywhere(deps: CycleDeps) -> dict[str, int | str]:
    """Cancel every open order at every venue; return {venue: count, or the error's type name}.

    A failed venue pages and never stops the others.
    """
    cancelled: dict[str, int | str] = {}
    for broker in deps.brokers:
        try:
            cancelled[broker.name] = broker.cancel_all_orders()
            log.info("killswitch_cancelled", broker=broker.name, count=cancelled[broker.name])
        except Exception as e:
            log.exception("killswitch_cancel_failed", broker=broker.name)
            cancelled[broker.name] = type(e).__name__
    return cancelled


def _submit(
    deps: CycleDeps,
    kind: str,
    order: ApprovedOrder,
    client_order_id: str,
    failed_brokers: set[str],
    now: datetime,
    mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Route one order by asset class, submit and book it; return (result fields, fills).

    Buys at an unreconciled venue are withheld; sells are not, since a full close
    never depends on the book. A filled entry left without its stop is still booked.
    """
    broker = _broker_for(deps, order.asset_class)
    if broker is None:
        return {"status": "no_broker", "detail": f"no broker for {order.asset_class}"}, []
    if broker.name in failed_brokers and order.side == "buy":
        return {"status": "reconcile_failed", "detail": f"{broker.name} book may be stale"}, []

    try:
        submitted = _submit_with_one_retry(broker, order, client_order_id)
    except (WrongVenueError, UnfillableQuantityError) as e:
        log.error("order_rejected_permanent", symbol=order.symbol, error=str(e))
        return {"status": "rejected_permanent", "detail": str(e)}, []
    except UnprotectedPositionError as e:
        log.error("position_unprotected", symbol=order.symbol, error=str(e))
        fills: list[dict[str, object]] = []
        if e.broker_order_id is not None:
            # The entry FILLED: the naked position must exist in our book.
            naked = SubmittedOrder(
                broker_order_id=e.broker_order_id,
                status="unprotected",  # non-terminal: polled until its fill is booked
                submitted_quantity=order.quantity,
            )
            fills = _book(deps, broker, order, naked, now, mode)
        return {"status": "unprotected_position", "detail": str(e)}, fills
    except BrokerError as e:
        log.error("order_failed", symbol=order.symbol, error=str(e))
        return {"status": "failed", "detail": str(e)}, []

    fills = _book(deps, broker, order, submitted, now, mode)
    return {
        "status": submitted.status,
        "broker_order_id": submitted.broker_order_id,
        "protective_order_id": submitted.protective_order_id,
    }, fills


def _submit_with_one_retry(
    broker: Broker, order: ApprovedOrder, client_order_id: str
) -> SubmittedOrder:
    """Submit an order, retrying once with the same key on a transient BrokerError.

    Permanent errors and UnprotectedPositionError are never retried: a blind resubmit
    of an unprotected fill would double the position.
    """
    try:
        return broker.submit_order(order, client_order_id)
    except (WrongVenueError, UnfillableQuantityError, UnprotectedPositionError):
        raise
    except BrokerError as e:
        log.warning("order_retrying", symbol=order.symbol, error=str(e))
        time.sleep(RETRY_DELAY_SECONDS)
        return broker.submit_order(order, client_order_id)


def _book(
    deps: CycleDeps,
    broker: Broker,
    order: ApprovedOrder,
    submitted: SubmittedOrder,
    now: datetime,
    mode: str,
) -> list[dict[str, object]]:
    """Write orders rows for a submission, then reconcile them; return the fills booked.

    Two transactions: the venue order is real, so its rows must survive the lookup.
    A failed write pages instead of raising, so the cycle still audits it.
    """
    try:
        with deps.session_factory() as session:
            orders = OrderRepository(session)
            row_ids = [
                orders.ensure(
                    broker=broker.name,
                    broker_order_id=submitted.broker_order_id,
                    created_at=now,
                    symbol=order.symbol,
                    asset_class=order.asset_class,
                    side=order.side,
                    quantity=submitted.submitted_quantity,
                    order_type="market",
                    stop_loss_price=order.stop_loss_price,
                    status=submitted.status,
                    trading_mode=mode,
                ).id
            ]
            if submitted.protective_order_id is not None:
                row_ids.append(
                    orders.ensure(
                        broker=broker.name,
                        broker_order_id=submitted.protective_order_id,
                        created_at=now,
                        symbol=order.symbol,
                        asset_class=order.asset_class,
                        side="sell",
                        quantity=submitted.submitted_quantity,
                        order_type="stop",
                        stop_loss_price=order.stop_loss_price,
                        status="open",
                        trading_mode=mode,
                    ).id
                )
            session.commit()
    except Exception:
        log.exception(
            "submission_unbooked",
            broker=broker.name,
            symbol=order.symbol,
            broker_order_id=submitted.broker_order_id,
            note="venue order exists but is NOT in the orders/trades book",
        )
        return []

    try:
        with deps.session_factory() as session:
            rows = [session.get(Order, row_id) for row_id in row_ids]
            booked, _ = _reconcile_rows(session, broker, rows, mode)
            session.commit()
            return booked
    except Exception:
        log.exception("immediate_reconcile_failed", broker=broker.name, symbol=order.symbol)
        return []


def reconcile_open_orders(deps: CycleDeps, mode: str) -> tuple[list[dict[str, object]], set[str]]:
    """Poll every open orders row at its venue; return (fills booked, venues with a failed lookup).

    Venue-side stop fills reach the book only this way.
    """
    booked: list[dict[str, object]] = []
    failed: set[str] = set()
    for broker in deps.brokers:
        with deps.session_factory() as session:
            rows = OrderRepository(session).open_orders(broker.name, mode)
            fills, ok = _reconcile_rows(session, broker, rows, mode)
            session.commit()  # keep what was booked even if a lookup failed
        booked += fills
        if not ok:
            failed.add(broker.name)
    return booked, failed


def _reconcile_rows(
    session: Session, broker: Broker, rows: list[Order], mode: str
) -> tuple[list[dict[str, object]], bool]:
    """Look up each row at the venue, book fills and retire terminal rows; (fills, all ok).

    Rows go oldest first so an entry's fill is booked before its stop's, which the stop's
    cost basis needs. A row the venue does not know, or a fill with no price, is retired
    as "lost" (paged) so one bad row cannot lock the venue out of submissions forever.
    The caller commits.
    """
    trades = TradeRepository(session)
    orders = OrderRepository(session)
    since = CashFlowRepository(session).start(mode)
    booked: list[dict[str, object]] = []
    ok = True
    for row in rows:
        try:
            venue = broker.get_order(row.broker_order_id)
        except UnknownOrderError:
            log.error("order_lost_at_venue", broker=broker.name, order_id=row.broker_order_id)
            row.status = "lost"
            continue
        except BrokerError as e:
            log.error("order_lookup_failed", broker=broker.name, order_id=row.broker_order_id)
            log.debug("order_lookup_error", error=str(e))
            ok = False
            continue
        if not venue.terminal:
            row.status = venue.status
            continue
        if venue.filled_quantity <= 0:
            row.status = "cancelled"
            continue
        if venue.fill_price is None:
            log.error("fill_unbookable_no_price", broker=broker.name, order_id=row.broker_order_id)
            row.status = "lost"
            continue
        if orders.has_fill(row.id):
            row.status = "filled"
            continue
        filled_at = venue.filled_at
        if filled_at is None:
            log.warning("fill_time_missing_using_now", broker=broker.name, symbol=row.symbol)
            filled_at = datetime.now(UTC)
        pnl = None
        if row.side == "sell":
            pnl = realized_pnl(
                trades.fills_for_symbol(row.symbol, mode, since),
                venue.fill_price,
                venue.filled_quantity,
                venue.fees,
            )
            if pnl is None:
                log.warning("realized_pnl_unknown_basis", symbol=row.symbol)
        trades.record_fill(
            filled_at=filled_at,
            symbol=row.symbol,
            asset_class=row.asset_class,
            side=row.side,
            quantity=venue.filled_quantity,
            fill_price=venue.fill_price,
            broker=broker.name,
            trading_mode=mode,
            fees=venue.fees,
            realized_pnl=pnl,
            order_id=row.id,
        )
        row.status = "filled"
        fill = {
            "symbol": row.symbol,
            "side": row.side,
            "order_type": row.order_type,
            "quantity": str(venue.filled_quantity),
            "fill_price": str(venue.fill_price),
            "realized_pnl": str(pnl) if pnl is not None else None,
            "broker_order_id": row.broker_order_id,
        }
        log.info("fill_booked", broker=broker.name, **fill)
        booked.append(fill)
    return booked, ok


def realized_pnl(
    prior_fills: list[Trade], fill_price: Decimal, quantity: Decimal, fees: Decimal
) -> Decimal | None:
    """Realized P&L of a closing fill against the average cost of prior_fills (oldest first).

    None when there is no open basis, so the streak breaker counts it as neither win nor loss.
    """
    _, avg_entry, _ = average_cost_walk(prior_fills)
    if avg_entry is None:
        return None
    return (fill_price - avg_entry) * quantity - fees


def _broker_for(deps: CycleDeps, asset_class: str) -> Broker | None:
    """The first broker declaring an asset class, or None when none is configured."""
    return next((b for b in deps.brokers if b.asset_class == asset_class), None)
