"""Layer-5 macro-event pause around FOMC, CPI and NFP releases.

macro_calendar.json holds release times in America/New_York; zoneinfo converts them to
UTC so DST comes from the tz database. A calendar that has run out looks exactly like
"nothing scheduled", so stale_kinds() lets the risk node warn once any kind runs out.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from zoneinfo import ZoneInfo

from trading_agent.risk.limits import RiskLimits

NEW_YORK = ZoneInfo("America/New_York")
KINDS = frozenset({"FOMC", "CPI", "NFP"})


@dataclass(frozen=True)
class MacroEvent:
    kind: str  # one of KINDS
    at: datetime  # release instant, UTC


def load_calendar(text: str) -> list[MacroEvent]:
    """Parse the calendar JSON into UTC events, sorted by release time.

    An unknown kind or bad date raises ValueError: a broken calendar must fail the
    import, never load as "no events".
    """
    events = []
    for row in json.loads(text)["events"]:
        if row["kind"] not in KINDS:
            raise ValueError(f"unknown macro event kind {row['kind']!r}")
        local = datetime.fromisoformat(f"{row['date']}T{row['time']}").replace(tzinfo=NEW_YORK)
        events.append(MacroEvent(kind=row["kind"], at=local.astimezone(UTC)))
    return sorted(events, key=lambda e: e.at)


def macro_pause(now: datetime, events: list[MacroEvent], limits: RiskLimits) -> bool:
    """Whether `now` falls inside any event's pause window, both ends inclusive.

    A naive `now` raises ValueError rather than silently comparing as benign.
    """
    if now.tzinfo is None:
        raise ValueError("macro_pause needs an aware datetime")
    before = timedelta(minutes=limits.macro_pause_before_minutes)
    after = timedelta(minutes=limits.macro_pause_after_minutes)
    return any(e.at - before <= now <= e.at + after for e in events)


def stale_kinds(now: datetime, events: list[MacroEvent]) -> list[str]:
    """Sorted event kinds with no release scheduled at or after `now`."""
    upcoming = {e.kind for e in events if e.at >= now}
    return sorted(KINDS - upcoming)


EVENTS: list[MacroEvent] = load_calendar(
    files("trading_agent.risk").joinpath("macro_calendar.json").read_text(encoding="utf-8")
)
