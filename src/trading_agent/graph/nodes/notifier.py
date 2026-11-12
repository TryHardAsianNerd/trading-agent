"""Node 7, notifier: cycle summary to the log, and to Discord when something happened.

The alert leads with every error-level event logged anywhere in the cycle, then
halted submissions, fills, closes, failures and stop breaches. Quiet cycles send
nothing, and a failing Discord never fails the cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trading_agent.graph.state import AgentState
from trading_agent.log import drain_alerts, get_logger
from trading_agent.notify.discord import format_cycle_alert
from trading_agent.risk.killswitch import scheduler_paused

if TYPE_CHECKING:
    from trading_agent.graph.deps import CycleDeps

log = get_logger(__name__)

# Error kinds paged since the scheduler's current pause began, seeded by the page that
# started it. Process-local: a restart clears the scheduler's pauses too.
_PAGED_KINDS: set[str] = set()


def hold_repeats(errors: list[dict], paused: bool) -> list[dict]:
    """The error events to page: during a scheduler pause, only kinds not yet paged in it."""
    kinds = {e.get("event") for e in errors}
    if not paused:
        _PAGED_KINDS.clear()
        _PAGED_KINDS.update(kinds)
        return errors
    held = [e.get("event") for e in errors if e.get("event") in _PAGED_KINDS]
    if held:
        log.warning("error_pages_held", held=held)
    new = [e for e in errors if e.get("event") not in _PAGED_KINDS]
    _PAGED_KINDS.update(kinds)
    return new


def notify(state: AgentState, *, deps: CycleDeps) -> dict:
    """Log the cycle summary and post the Discord alert when warranted."""
    log.info(
        "cycle_summary",
        cycle_id=state.get("cycle_id"),
        proposals=len(state.get("proposals") or []),
        approved=len(state.get("approved_orders") or []),
        rejected=len(state.get("rejections") or []),
        stop_breaches=len(state.get("stop_breaches") or []),
        executed=len(state.get("execution_results") or []),
    )
    sent = ["log"]
    errors = drain_alerts()  # always drained: a quiet cycle must not carry errors forward
    if deps.notifier is None:
        return {"notifications_sent": sent}
    errors = hold_repeats(errors, scheduler_paused(deps.kill_switch.is_halted()[1]))
    text = format_cycle_alert(state, errors, deps.settings.trading_mode)
    if text is not None and deps.notifier.send(text):
        sent.append("discord")
    return {"notifications_sent": sent}
