"""Scheduler tests: halt policy, market hours, watchdog, resume, overlap, serve.

Hermetic: a fake cycle callable, a fake NYSE predicate, FakeRedis behind the
real KillSwitch, and a recording notifier. The pruning query is covered in
test_graph_integration (needs the checkpoint tables).
"""

import asyncio
import threading
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from redis import RedisError
from tests.fakes import FakeRedis

from trading_agent.config import Settings
from trading_agent.log import configure_logging, drain_alerts
from trading_agent.risk.killswitch import COOLDOWN_KEY, KILL_SWITCH_KEY, KillSwitch
from trading_agent.risk.limits import DEFAULT_LIMITS
from trading_agent.scheduler import (
    CONNECTIVITY_PAUSE_REASON,
    CRASH_PAUSE_REASON,
    IN_FLIGHT_KEY,
    IN_FLIGHT_TTL_SECONDS,
    AgentRunner,
    connectivity_problem,
    cycle_trigger,
    nyse_open,
    serve,
)


@pytest.fixture(autouse=True)
def _fresh_alert_buffer():
    """Other modules log error events without draining; start each test clean.

    Also installs the production logging chain: error events reach the alert
    buffer only through log.collect_alerts, which configure_logging wires in.
    Without it these tests would pass only when an earlier module happened to
    configure logging.
    """
    configure_logging()
    drain_alerts()
    yield
    drain_alerts()


class RecordingNotifier:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return True


class FakeCycle:
    """Records (stocks_open, cycle_id) per call; returns a canned final state."""

    def __init__(self, final=None, raises=None):
        self.calls = []
        self.final = final if final is not None else {"cycle_id": "c"}
        self.raises = raises  # exception, or a list consumed one per call (None = success)
        self.gate = None  # threading.Event to hold the cycle open (overlap test)

    def __call__(self, stocks_open, cycle_id):
        self.calls.append((stocks_open, cycle_id))
        if self.gate is not None:
            self.gate.wait()
        boom = self.raises.pop(0) if isinstance(self.raises, list) else self.raises
        if boom is not None:
            raise boom
        return self.final


def _runner(cycle, redis=None, notifier=None, stocks="AAPL", market_open=lambda now: True):
    settings = Settings(_env_file=None, stock_watchlist=stocks, crypto_watchlist="BTC-USD")
    deps = SimpleNamespace(kill_switch=KillSwitch(redis or FakeRedis()), notifier=notifier)
    return AgentRunner(settings, deps, cycle, market_open=market_open)


# ------------------------------------------------------------ market hours


def test_cycle_runs_with_the_market_flag_and_a_fresh_cycle_id():
    cycle = FakeCycle()
    assert _runner(cycle, market_open=lambda now: False).run_once() == "ran"
    assert _runner(cycle, market_open=lambda now: True).run_once() == "ran"
    assert [c[0] for c in cycle.calls] == [False, True]
    assert cycle.calls[0][1] != cycle.calls[1][1]


def test_no_stock_watchlist_never_consults_the_calendar():
    cycle, calls = FakeCycle(), []
    _runner(cycle, stocks="", market_open=lambda now: calls.append(now) or True).run_once()
    assert cycle.calls[0][0] is False and calls == []


def test_calendar_failure_fails_closed_to_no_stocks():
    cycle = FakeCycle()

    def broken(now):
        raise RuntimeError("calendar out of range")

    assert _runner(cycle, market_open=broken).run_once() == "ran"
    assert cycle.calls[0][0] is False


# ------------------------------------------------------------ halt policy


@pytest.mark.parametrize("pause", [CRASH_PAUSE_REASON, CONNECTIVITY_PAUSE_REASON])
def test_the_runners_own_pauses_still_run_the_cycle(pause):
    """A pause stops new entries only: the core's trail and stop re-protection need the cycle."""
    redis, notifier, cycle = FakeRedis(), RecordingNotifier(), FakeCycle()
    KillSwitch(redis).start_cooldown(pause, DEFAULT_LIMITS)
    assert _runner(cycle, redis, notifier).run_once() == "ran"
    assert len(cycle.calls) == 1 and notifier.sent == []


def test_kill_switch_skips_the_cycle_and_notifies_once_per_halt():
    redis, notifier, cycle = FakeRedis(), RecordingNotifier(), FakeCycle()
    runner = _runner(cycle, redis, notifier)
    KillSwitch(redis).activate("manual_discord")

    assert runner.run_once() == "skipped_halted"
    assert runner.run_once() == "skipped_halted"
    assert cycle.calls == []
    assert notifier.sent == [
        "[paper] HALTED (kill_switch:manual_discord) - cycles skipped until cleared"
    ]

    KillSwitch(redis).deactivate_human_only("jason")
    assert runner.run_once() == "ran"
    assert notifier.sent[-1] == "[paper] Kill switch cleared - cycles resumed"
    assert runner.run_once() == "ran"
    assert len(notifier.sent) == 2  # no repeat of the resume notice


def test_gate_breaker_cooldown_still_runs_the_cycle_silently():
    """The gate rejects and the executor refuses under its own cooldown, but
    snapshots, reconciliation and stop settlement must continue — and the
    runner adds no HALTED/cleared chatter (the breaker trip already paged)."""
    redis, notifier, cycle = FakeRedis(), RecordingNotifier(), FakeCycle()
    KillSwitch(redis).start_cooldown("daily_loss_breaker", DEFAULT_LIMITS)
    runner = _runner(cycle, redis, notifier)
    assert runner.run_once() == "ran"
    assert len(cycle.calls) == 1 and notifier.sent == []


# -------------------------------------------------------------- watchdog


def test_crash_pages_pauses_and_resumes_the_same_cycle_once():
    redis, notifier = FakeRedis(), RecordingNotifier()
    cycle = FakeCycle(raises=[RuntimeError("postgresql://u:secret@h"), None])
    runner = _runner(cycle, redis, notifier)

    assert runner.run_once() == "crashed"
    assert notifier.sent[0].splitlines() == [
        "[paper] CYCLE CRASHED: RuntimeError",
        "ERROR cycle_crashed",
        "ERROR breaker_cooldown_started reason=cycle_crashed",  # paused BEFORE the drain
    ]
    assert "secret" not in notifier.sent[0]
    assert drain_alerts() == []  # nothing left to page an hour late
    assert redis.store[COOLDOWN_KEY] == CRASH_PAUSE_REASON
    assert redis.ttls[COOLDOWN_KEY] >= 3600
    assert KILL_SWITCH_KEY not in redis.store  # a pause, not the durable switch

    # The pause stops new entries only: the next run resumes the crashed cycle under
    # the same id, with no extra notice.
    assert runner.run_once() == "ran"
    assert cycle.calls[0][1] == cycle.calls[1][1]
    assert len(notifier.sent) == 1
    # A later cycle gets a fresh id.
    runner.run_once()
    assert cycle.calls[2][1] != cycle.calls[1][1]


def test_a_cycle_that_crashes_twice_is_abandoned():
    redis = FakeRedis()
    cycle = FakeCycle(raises=[RuntimeError("a"), RuntimeError("b"), None])
    runner = _runner(cycle, redis)
    runner.run_once()
    redis.delete(COOLDOWN_KEY)
    runner.run_once()  # resume attempt crashes too
    redis.delete(COOLDOWN_KEY)
    runner.run_once()
    ids = [c[1] for c in cycle.calls]
    assert ids[0] == ids[1] and ids[2] != ids[1]

    assert IN_FLIGHT_KEY not in redis.store


def test_a_finished_cycle_leaves_no_marker_and_a_crash_leaves_its_id():
    redis = FakeRedis()
    cycle = FakeCycle(raises=[None, RuntimeError("x")])
    runner = _runner(cycle, redis)
    runner.run_once()
    assert IN_FLIGHT_KEY not in redis.store
    runner.run_once()
    assert redis.store[IN_FLIGHT_KEY] == cycle.calls[1][1]
    assert redis.ttls[IN_FLIGHT_KEY] == IN_FLIGHT_TTL_SECONDS


def test_a_cycle_cut_short_by_the_process_dying_is_resumed_on_restart():
    """The marker is in Redis, so a NEW runner (a restarted process) replays
    the unfinished cycle under its own id - its submissions book under the
    same idempotency keys instead of surfacing as untracked positions."""
    redis = FakeRedis()
    redis.set(IN_FLIGHT_KEY, "died-mid-cycle")
    cycle = FakeCycle()
    assert _runner(cycle, redis).run_once() == "ran"
    assert cycle.calls[0][1] == "died-mid-cycle"
    assert IN_FLIGHT_KEY not in redis.store


def test_a_cycle_that_kills_the_process_every_time_is_replayed_only_once():
    """Two restarts in a row die inside the same cycle: the second restart
    must abandon it (paged), not replay it until the marker expires."""
    redis = FakeRedis()
    redis.set(IN_FLIGHT_KEY, "kills-the-process")
    first = _runner(FakeCycle(), redis)
    assert first._claim_cycle() == ("kills-the-process", True)  # ...and the process dies
    assert redis.store[IN_FLIGHT_KEY] == "kills-the-process:resumed"
    assert redis.ttls[IN_FLIGHT_KEY] == IN_FLIGHT_TTL_SECONDS

    cycle = FakeCycle()
    _runner(cycle, redis).run_once()
    assert cycle.calls[0][1] != "kills-the-process"
    assert [a["event"] for a in drain_alerts()] == ["cycle_abandoned"]


def test_the_production_client_returns_bytes():
    """redis-py without decode_responses (graph/deps.py) returns bytes."""

    class BytesRedis(FakeRedis):
        def get(self, key):
            value = super().get(key)
            return value.encode() if isinstance(value, str) else value

    redis = BytesRedis()
    redis.set(IN_FLIGHT_KEY, "crashed-cycle")
    cycle = FakeCycle()
    assert _runner(cycle, redis).run_once() == "ran"
    assert cycle.calls[0][1] == "crashed-cycle"


def test_a_failed_marker_delete_never_stops_the_cycle():
    class UndeletableRedis(FakeRedis):
        def delete(self, key):
            if key == IN_FLIGHT_KEY:
                raise RedisError("timeout")
            super().delete(key)

    assert _runner(FakeCycle(), UndeletableRedis()).run_once() == "ran"


def test_a_replay_cut_short_too_is_abandoned_and_paged():
    redis = FakeRedis()
    redis.set(IN_FLIGHT_KEY, "died-mid-cycle:resumed")
    cycle = FakeCycle()
    _runner(cycle, redis).run_once()
    assert cycle.calls[0][1] != "died-mid-cycle"
    assert [a["event"] for a in drain_alerts()] == ["cycle_abandoned"]


def test_without_the_marker_the_cycle_still_runs_on_a_fresh_id():
    """Redis answering the halt check but failing the marker must not stop trading."""

    class MarkerlessRedis(FakeRedis):
        def get(self, key):
            if key == IN_FLIGHT_KEY:
                raise RedisError("timeout")
            return super().get(key)

    cycle = FakeCycle()
    assert _runner(cycle, MarkerlessRedis()).run_once() == "ran"
    assert len(cycle.calls) == 1


def test_connectivity_loss_pauses_and_pages_but_never_cancels():
    redis, notifier = FakeRedis(), RecordingNotifier()
    final = {"cycle_id": "c", "portfolio_error": "BrokerError", "proposals": []}
    runner = _runner(FakeCycle(final=final), redis, notifier)

    assert runner.run_once() == "ran"

    assert redis.store[COOLDOWN_KEY] == CONNECTIVITY_PAUSE_REASON
    (msg,) = notifier.sent
    assert msg.startswith("[paper] CONNECTIVITY LOST: portfolio assembly failed (BrokerError)")
    assert "the core and protective stops keep running" in msg
    assert msg.splitlines()[-1] == "ERROR breaker_cooldown_started reason=connectivity_lost"
    assert drain_alerts() == []
    assert KILL_SWITCH_KEY not in redis.store


def test_a_pause_that_lifted_during_the_cycle_is_armed_again_by_its_crash():
    """The pause is checked when the crash happens, not when the cycle started."""
    redis, notifier = FakeRedis(), RecordingNotifier()
    KillSwitch(redis).start_cooldown(CRASH_PAUSE_REASON, DEFAULT_LIMITS)
    drain_alerts()

    class LiftsThenCrashes(FakeCycle):
        def __call__(self, stocks_open, cycle_id):
            redis.delete(COOLDOWN_KEY)  # the hour runs out mid-cycle
            return super().__call__(stocks_open, cycle_id)

    runner = _runner(LiftsThenCrashes(raises=RuntimeError("again")), redis, notifier)
    assert runner.run_once() == "crashed"
    assert redis.store[COOLDOWN_KEY] == CRASH_PAUSE_REASON
    assert redis.ttls[COOLDOWN_KEY] >= 3600


def test_an_active_pause_is_not_rearmed_or_reannounced():
    """While a pause runs, a repeat crash and an ongoing outage page nothing and leave
    the pause to lift on time."""
    redis, notifier = FakeRedis(), RecordingNotifier()
    crashes = [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")]
    runner = _runner(FakeCycle(raises=crashes), redis, notifier)
    assert runner.run_once() == "crashed"  # starts the pause and pages
    redis.ttls[COOLDOWN_KEY] = 600  # ten minutes left
    assert runner.run_once() == "crashed"  # the replay crashes: only the abandonment is new
    assert runner.run_once() == "crashed"  # a fresh cycle crashes: nothing new, nothing paged
    assert redis.ttls[COOLDOWN_KEY] == 600
    assert len(notifier.sent) == 2 and drain_alerts() == []
    assert "cycle_abandoned" in notifier.sent[1] and "cycle_crashed" not in notifier.sent[1]

    redis, notifier = FakeRedis(), RecordingNotifier()
    KillSwitch(redis).start_cooldown(CONNECTIVITY_PAUSE_REASON, DEFAULT_LIMITS)
    redis.ttls[COOLDOWN_KEY] = 600
    drain_alerts()
    final = {"cycle_id": "c", "reconcile_failed": ["alpaca"]}
    assert _runner(FakeCycle(final=final), redis, notifier).run_once() == "ran"
    assert redis.ttls[COOLDOWN_KEY] == 600 and notifier.sent == []


def test_connectivity_problem_detection_is_independent_of_proposals():
    """A venue outage on a holds-only cycle (the common case) must be seen."""
    assert connectivity_problem({"cycle_id": "c"}) is None
    assert connectivity_problem({"rejections": [{"reason": "evaluation_error"}]}) is None
    assert (
        connectivity_problem({"reconcile_failed": ["alpaca"]})
        == "venue order lookups failed (alpaca)"
    )
    assert (
        connectivity_problem({"portfolio_error": "OperationalError", "proposals": []})
        == "portfolio assembly failed (OperationalError)"
    )


def test_clean_cycle_neither_pauses_nor_pages():
    redis, notifier = FakeRedis(), RecordingNotifier()
    final = {
        "cycle_id": "c",
        "execution_results": [{"status": "filled"}],
        "rejections": [{"reason": "no_position_to_sell"}],
        "reconcile_failed": [],
    }
    assert _runner(FakeCycle(final=final), redis, notifier).run_once() == "ran"
    assert notifier.sent == [] and COOLDOWN_KEY not in redis.store


def test_runner_without_notifier_still_runs():
    redis = FakeRedis()
    assert (
        _runner(FakeCycle(raises=RuntimeError("x")), redis, notifier=None).run_once() == "crashed"
    )
    assert redis.store[COOLDOWN_KEY] == CRASH_PAUSE_REASON


# ------------------------------------------------------------------ tick


def test_tick_never_overlaps_a_running_cycle():
    cycle = FakeCycle()
    cycle.gate = threading.Event()
    runner = _runner(cycle)

    async def scenario():
        first = asyncio.create_task(runner.tick())
        await asyncio.sleep(0.05)  # first tick is inside the cycle, holding the lock
        second = await runner.tick()
        cycle.gate.set()
        return await first, second

    assert asyncio.run(scenario()) == ("ran", "skipped_running")
    assert len(cycle.calls) == 1


# --------------------------------------------------------------- serve()


def test_serve_runs_a_first_cycle_posts_notices_and_clears_its_own_pause():
    """Wiring test without Discord or a database: a fake graph stands in for
    the compiled one; cancellation drives the shutdown path."""
    redis, notifier = FakeRedis(), RecordingNotifier()
    KillSwitch(redis).start_cooldown(CRASH_PAUSE_REASON, DEFAULT_LIMITS)  # left by a crash
    invoked = []
    graph = SimpleNamespace(
        checkpointer=None, invoke=lambda initial, config: invoked.append(initial) or initial
    )
    settings = Settings(_env_file=None, stock_watchlist="", crypto_watchlist="BTC-USD")
    deps = SimpleNamespace(
        kill_switch=KillSwitch(redis), notifier=notifier, brokers=[], tracker=None
    )

    async def scenario():
        task = asyncio.create_task(serve(settings, deps, graph, engine=None))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert COOLDOWN_KEY not in redis.store  # restart cleared the runner's pause
    assert len(invoked) == 1 and invoked[0]["stock_market_open"] is False
    assert notifier.sent[0].startswith("[paper] agent started: every 15 min")
    assert notifier.sent[-1].startswith("[paper] agent stopping")


# --------------------------------------------------------- trigger/calendar


def test_cycle_trigger_aligns_to_the_hour_and_rejects_odd_intervals():
    trigger = cycle_trigger(15)
    nxt = trigger.get_next_fire_time(None, datetime(2026, 9, 21, 15, 1, tzinfo=UTC))
    assert (nxt.hour, nxt.minute) == (15, 15)
    with pytest.raises(ValueError):
        cycle_trigger(7)
    with pytest.raises(ValueError):
        cycle_trigger(0)


def test_nyse_open_knows_sessions_weekends_holidays_and_rejects_naive_input():
    assert nyse_open(datetime(2026, 9, 21, 15, 0, tzinfo=UTC))  # Monday 11:00 ET
    assert not nyse_open(datetime(2026, 9, 21, 12, 0, tzinfo=UTC))  # Monday 08:00 ET, pre-open
    assert not nyse_open(datetime(2026, 9, 19, 15, 0, tzinfo=UTC))  # Saturday
    assert not nyse_open(datetime(2026, 7, 3, 15, 0, tzinfo=UTC))  # Independence Day observed
    with pytest.raises(TypeError):
        nyse_open(datetime(2026, 9, 21, 15, 0))  # naive: the runner fails closed on this


def test_serve_enters_and_starts_the_bot_before_the_first_cycle(monkeypatch):
    """The kill switch must exist from the first trade: the bot is entered
    (discord.Client context manager), started, and ready before the first
    cycle runs; cancellation exits the context (closing the connection)."""
    import trading_agent.scheduler as scheduler_module

    events = []

    class FakeBot:
        """Mirrors discord.Client: wait_until_ready blocks until start() is running."""

        def __init__(self, token, channel_id, owner_id, kill_switch, brokers, tracker):
            events.append("built")
            self._ready = asyncio.Event()

        async def __aenter__(self):
            events.append("entered")
            return self

        async def __aexit__(self, *exc):
            events.append("closed")

        async def start(self, token):
            events.append("start")
            self._ready.set()
            await asyncio.Event().wait()

        async def wait_until_ready(self):
            await self._ready.wait()
            events.append("ready")

    monkeypatch.setattr(scheduler_module, "DiscordBot", FakeBot)
    invoked = []
    graph = SimpleNamespace(
        checkpointer=None,
        invoke=lambda initial, config: events.append("cycle") or invoked.append(initial) or initial,
    )
    settings = Settings(
        _env_file=None,
        stock_watchlist="",
        crypto_watchlist="BTC-USD",
        discord_bot_token="t",
        discord_channel_id="1",
        discord_owner_id="2",
    )
    deps = SimpleNamespace(
        kill_switch=KillSwitch(FakeRedis()), notifier=RecordingNotifier(), brokers=[], tracker=None
    )

    async def scenario():
        task = asyncio.create_task(serve(settings, deps, graph, engine=None))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert events == ["built", "entered", "start", "ready", "cycle", "closed"], events
    assert "commands on" in deps.notifier.sent[0]


def test_serve_never_runs_a_cycle_while_the_bot_gateway_is_not_ready(monkeypatch):
    """Fail closed: no kill switch reachable -> no cycle, only periodic warnings."""
    import trading_agent.scheduler as scheduler_module

    class NeverReadyBot:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def start(self, token):
            await asyncio.Event().wait()

        async def wait_until_ready(self):
            await asyncio.Event().wait()

    monkeypatch.setattr(scheduler_module, "DiscordBot", NeverReadyBot)
    monkeypatch.setattr(scheduler_module, "BOT_READY_TIMEOUT_SECONDS", 0.05)
    invoked = []
    graph = SimpleNamespace(
        checkpointer=None, invoke=lambda initial, config: invoked.append(initial) or initial
    )
    settings = Settings(
        _env_file=None,
        stock_watchlist="",
        crypto_watchlist="BTC-USD",
        discord_bot_token="t",
        discord_channel_id="1",
        discord_owner_id="2",
    )
    deps = SimpleNamespace(
        kill_switch=KillSwitch(FakeRedis()), notifier=RecordingNotifier(), brokers=[], tracker=None
    )

    async def scenario():
        task = asyncio.create_task(serve(settings, deps, graph, engine=None))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert invoked == [] and deps.notifier.sent == []
