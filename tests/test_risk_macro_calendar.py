"""Layer-5 macro-event pause: parsing, window boundaries, DST, staleness.

`risk/` is exhaustively tested by policy (D-008). The pause is a pure function,
so every case here is exact — no tolerance, no floats, no network.
"""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading_agent.risk import macro_calendar
from trading_agent.risk.limits import DEFAULT_LIMITS, RiskLimits
from trading_agent.risk.macro_calendar import (
    KINDS,
    NEW_YORK,
    MacroEvent,
    load_calendar,
    macro_pause,
    stale_kinds,
)

BEFORE = timedelta(minutes=DEFAULT_LIMITS.macro_pause_before_minutes)
AFTER = timedelta(minutes=DEFAULT_LIMITS.macro_pause_after_minutes)


def _events(*instants: datetime) -> list[MacroEvent]:
    """Wrap UTC instants as CPI events for window tests, in the given order."""
    return [MacroEvent(kind="CPI", at=i) for i in instants]


# --- parsing ---------------------------------------------------------------


def test_load_calendar_converts_new_york_clock_times_to_utc():
    """An 08:30 ET print in January (EST, UTC-5) is 13:30 UTC."""
    text = '{"events": [{"kind": "CPI", "date": "2026-01-13", "time": "08:30"}]}'
    (event,) = load_calendar(text)
    assert event.kind == "CPI"
    assert event.at == datetime(2026, 1, 13, 13, 30, tzinfo=UTC)


def test_load_calendar_is_dst_correct_not_fixed_offset():
    """The SAME 08:30 ET clock time is 13:30 UTC in EST and 12:30 UTC in EDT.
    A hardcoded offset would put the summer window an hour wrong — i.e. the
    gate unpaused through the actual release."""
    text = """{"events": [
        {"kind": "CPI", "date": "2026-01-13", "time": "08:30"},
        {"kind": "CPI", "date": "2026-07-14", "time": "08:30"}
    ]}"""
    winter, summer = load_calendar(text)
    assert winter.at == datetime(2026, 1, 13, 13, 30, tzinfo=UTC)
    assert summer.at == datetime(2026, 7, 14, 12, 30, tzinfo=UTC)
    assert winter.at.astimezone(NEW_YORK).hour == summer.at.astimezone(NEW_YORK).hour == 8


def test_load_calendar_sorts_by_instant():
    text = """{"events": [
        {"kind": "FOMC", "date": "2026-06-17", "time": "14:00"},
        {"kind": "CPI", "date": "2026-01-13", "time": "08:30"}
    ]}"""
    assert [e.kind for e in load_calendar(text)] == ["CPI", "FOMC"]


def test_load_calendar_rejects_unknown_kind():
    """A typo must fail the import, never load as a silently smaller calendar."""
    text = '{"events": [{"kind": "GDP", "date": "2026-01-13", "time": "08:30"}]}'
    with pytest.raises(ValueError, match="unknown macro event kind"):
        load_calendar(text)


def test_load_calendar_rejects_unparseable_time():
    text = '{"events": [{"kind": "CPI", "date": "2026-01-13", "time": "8.30am"}]}'
    with pytest.raises(ValueError):
        load_calendar(text)


# --- window boundaries -----------------------------------------------------


RELEASE = datetime(2026, 3, 11, 12, 30, tzinfo=UTC)  # CPI 08:30 ET, EDT


@pytest.mark.parametrize(
    ("label", "now", "expected"),
    [
        ("exact release instant", RELEASE, True),
        ("window start (inclusive)", RELEASE - BEFORE, True),
        ("window end (inclusive)", RELEASE + AFTER, True),
        ("one second before the window", RELEASE - BEFORE - timedelta(seconds=1), False),
        ("one second after the window", RELEASE + AFTER + timedelta(seconds=1), False),
        ("inside, pre-release", RELEASE - timedelta(minutes=5), True),
        ("inside, post-release", RELEASE + timedelta(minutes=45), True),
        ("same day, hours later", RELEASE + timedelta(hours=6), False),
        ("day before", RELEASE - timedelta(days=1), False),
    ],
)
def test_macro_pause_window_boundaries(label, now, expected):
    assert macro_pause(now, _events(RELEASE), DEFAULT_LIMITS) is expected, label


def test_macro_pause_is_false_for_an_event_on_another_day():
    other_day = datetime(2026, 3, 12, 12, 30, tzinfo=UTC)
    assert macro_pause(RELEASE, _events(other_day), DEFAULT_LIMITS) is False


def test_macro_pause_matches_any_event_not_just_the_first():
    later = RELEASE + timedelta(days=30)
    assert macro_pause(later, _events(RELEASE, later), DEFAULT_LIMITS) is True


def test_macro_pause_is_false_on_an_empty_calendar():
    """Documented trap: an exhausted calendar is indistinguishable from a calm
    market here. stale_kinds() is what makes it visible; see the node test."""
    assert macro_pause(RELEASE, [], DEFAULT_LIMITS) is False


def test_macro_pause_honours_the_configured_window():
    tight = RiskLimits(macro_pause_before_minutes=1, macro_pause_after_minutes=1)
    assert macro_pause(RELEASE - timedelta(minutes=5), _events(RELEASE), tight) is False
    assert macro_pause(RELEASE - timedelta(minutes=1), _events(RELEASE), tight) is True


def test_macro_pause_compares_instants_across_timezones():
    """`now` in any zone is the same instant: a NY-local now inside the window pauses."""
    ny_now = RELEASE.astimezone(NEW_YORK)
    assert macro_pause(ny_now, _events(RELEASE), DEFAULT_LIMITS) is True
    assert macro_pause(RELEASE.astimezone(ZoneInfo("Asia/Tokyo")), _events(RELEASE), DEFAULT_LIMITS)


def test_macro_pause_rejects_a_naive_now():
    """A naive/aware comparison raises TypeError deep inside `any()`; catching it
    upstream would read as 'no pause'. Fail loudly instead."""
    with pytest.raises(ValueError, match="aware datetime"):
        macro_pause(RELEASE.replace(tzinfo=None), _events(RELEASE), DEFAULT_LIMITS)


def test_macro_pause_spans_a_dst_transition_day():
    """2026-03-08 is the US spring-forward. An 08:30 ET print the following
    Wednesday is EDT; the window either side of it must still be wall-clock
    correct, i.e. 12:00-13:30 UTC, not 13:00-14:30."""
    (event,) = load_calendar('{"events": [{"kind": "CPI", "date": "2026-03-11", "time": "08:30"}]}')
    assert macro_pause(datetime(2026, 3, 11, 12, 0, tzinfo=UTC), [event], DEFAULT_LIMITS) is True
    assert macro_pause(datetime(2026, 3, 11, 13, 30, tzinfo=UTC), [event], DEFAULT_LIMITS) is True
    assert macro_pause(datetime(2026, 3, 11, 13, 31, tzinfo=UTC), [event], DEFAULT_LIMITS) is False
    # The pre-transition (EST) reading of the same clock time would be an hour later.
    assert macro_pause(datetime(2026, 3, 11, 14, 0, tzinfo=UTC), [event], DEFAULT_LIMITS) is False


# --- staleness -------------------------------------------------------------


def test_stale_kinds_lists_kinds_with_nothing_upcoming():
    events = [MacroEvent(kind="FOMC", at=RELEASE)]
    assert stale_kinds(RELEASE - timedelta(days=1), events) == ["CPI", "NFP"]


def test_stale_kinds_counts_an_event_at_now_as_upcoming():
    events = [MacroEvent(kind=k, at=RELEASE) for k in sorted(KINDS)]
    assert stale_kinds(RELEASE, events) == []


def test_stale_kinds_returns_every_kind_for_an_empty_calendar():
    assert stale_kinds(RELEASE, []) == sorted(KINDS)


# --- the shipped calendar --------------------------------------------------


def test_shipped_calendar_has_every_kind_and_no_duplicates():
    events = macro_calendar.EVENTS
    assert {e.kind for e in events} == set(KINDS)
    assert len({(e.kind, e.at) for e in events}) == len(events)
    assert events == sorted(events, key=lambda e: e.at)


def test_shipped_calendar_is_not_about_to_run_out():
    """The maintenance trap, guarded in the only way that works: RELATIVE to
    today. A hardcoded date would keep passing forever after the calendar
    expired — and the expired state is silent at the call site (no upcoming
    event reads exactly like a calm market), so this test is the thing that
    has to go red. It does so 30 days before the last release on file."""
    soon = datetime.now(UTC) + timedelta(days=30)
    assert stale_kinds(soon, macro_calendar.EVENTS) == [], (
        "the shipped macro calendar runs out within 30 days — add the next year's "
        "FOMC/CPI/NFP rows to macro_calendar.json (sources cited in the file)"
    )


def test_shipped_bls_releases_are_at_0830_new_york():
    for event in macro_calendar.EVENTS:
        local = event.at.astimezone(NEW_YORK)
        expected = (8, 30) if event.kind in {"CPI", "NFP"} else (14, 0)
        assert (local.hour, local.minute) == expected, f"{event.kind} {local}"
        assert local.weekday() < 5, f"{event.kind} {local} falls on a weekend"
