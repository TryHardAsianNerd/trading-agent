"""The long-running agent: quarter-hour cycles, the Discord command bot and checkpoint pruning.

With the bot configured, no cycle runs until it is ready, so the phone kill switch exists
before any trade. The manual kill switch, an unreachable Redis or an unknown halt reason
skips cycles; other halts let them run with swing orders stopped. A connectivity failure or
crash pauses new entries for an hour and pages, but never cancels the protective stops. A
crashed cycle is resumed once, even after a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import exchange_calendars
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from langgraph.graph.state import CompiledStateGraph
from redis import RedisError
from sqlalchemy import Engine

from trading_agent.config import Settings
from trading_agent.graph.builder import run_cycle
from trading_agent.graph.deps import CycleDeps
from trading_agent.graph.nodes.notifier import hold_repeats
from trading_agent.graph.state import AgentState
from trading_agent.log import drain_alerts, get_logger
from trading_agent.notify.discord import DiscordBot, error_lines, format_crash_alert
from trading_agent.risk.killswitch import (
    CONNECTIVITY_PAUSE_REASON,
    CRASH_PAUSE_REASON,
    RUNNER_PAUSE_REASONS,
    stops_everything,
)
from trading_agent.risk.limits import DEFAULT_LIMITS
from trading_agent.storage.db import prune_checkpoints

log = get_logger(__name__)

PRUNE_AFTER_DAYS = 30
# A slot that could not start within this window is skipped, not replayed.
MISFIRE_GRACE_SECONDS = 60
# How often startup logs that it is still waiting for the Discord gateway.
BOT_READY_TIMEOUT_SECONDS = 30
# "<id>" during a cycle's first run (or after it crashed), "<id>:resumed" during its replay.
IN_FLIGHT_KEY = "agent:cycle_in_flight"
RESUMED_SUFFIX = ":resumed"
# Outlives the 1h crash pause; an older cycle is not replayed (its orders were sized on old prices).
IN_FLIGHT_TTL_SECONDS = 2 * 3600


def nyse_open(now: datetime) -> bool:
    """Whether the NYSE regular session is open at a timezone-aware instant."""
    calendar = exchange_calendars.get_calendar("XNYS")
    return bool(calendar.is_open_on_minute(pd.Timestamp(now).tz_convert("UTC")))


def stocks_open(settings: Settings, market_open: Callable[[datetime], bool] = nyse_open) -> bool:
    """Whether to take stock quotes this cycle; a calendar failure fails closed. Never raises."""
    if not settings.stock_symbols:
        return False
    try:
        return market_open(datetime.now(UTC))
    except Exception:
        log.warning("market_calendar_failed", exc_info=True)
        return False


def connectivity_problem(final: AgentState) -> str | None:
    """Why the cycle could not trust its view of a venue or the book, or None."""
    if final.get("portfolio_error"):
        return f"portfolio assembly failed ({final['portfolio_error']})"
    if final.get("reconcile_failed"):
        return f"venue order lookups failed ({', '.join(final['reconcile_failed'])})"
    return None


class AgentRunner:
    """Runs cycles on demand with the halt check, market hours and watchdog."""

    def __init__(
        self,
        settings: Settings,
        deps: CycleDeps,
        run: Callable[[bool, str], AgentState],
        market_open: Callable[[datetime], bool] = nyse_open,
    ) -> None:
        """Wire the runner; `run` runs one cycle from (stock_market_open, cycle_id)."""
        self._settings = settings
        self._deps = deps
        self._run = run
        self._market_open = market_open
        self._lock = asyncio.Lock()
        self._kill_notified = False

    async def tick(self) -> str:
        """Run one cycle off the event loop, or return "skipped_running" if one is in progress."""
        if self._lock.locked():
            log.warning("cycle_overlap_skipped")
            return "skipped_running"
        async with self._lock:
            return await asyncio.to_thread(self.run_once)

    def run_once(self) -> str:
        """Halt check, cycle, watchdog: "ran", "skipped_halted" or "crashed". Never raises."""
        halted, reason = self._deps.kill_switch.is_halted()
        if stops_everything(halted, reason):
            log.warning("cycle_skipped_halted", reason=reason)
            if not self._kill_notified:
                self.notify(f"HALTED ({reason}) - cycles skipped until cleared")
                self._kill_notified = True
            return "skipped_halted"
        if self._kill_notified:
            self.notify("Kill switch cleared - cycles resumed")
            self._kill_notified = False

        cycle_id, resumed = self._claim_cycle()
        try:
            final = self._run(stocks_open(self._settings, self._market_open), cycle_id)
        except Exception as e:
            log.exception("cycle_crashed", cycle_id=cycle_id, resumed=resumed)
            if resumed:
                self._release_cycle()
                log.error("cycle_abandoned", cycle_id=cycle_id, note="crashed twice; not resumed")
            # Otherwise the marker stays and the unfinished nodes replay next run.
            if self._paused_for(CRASH_PAUSE_REASON):
                # This pause already paged its crash: page only kinds of error new to it.
                alerts = hold_repeats(drain_alerts(), paused=True)
            else:
                # Pause FIRST: its error-level event must ride THIS alert, not the
                # next cycle's (which would page it an hour late as fresh news).
                self._pause(CRASH_PAUSE_REASON)
                alerts = hold_repeats(drain_alerts(), paused=False)
            if alerts and self._deps.notifier is not None:
                self._deps.notifier.send(format_crash_alert(e, alerts, self._settings.trading_mode))
            return "crashed"
        self._release_cycle()
        problem = connectivity_problem(final)
        # An active pause is not re-armed or re-announced: it lifts after an hour,
        # and a problem that persists starts a new one (hourly pages, not one per cycle).
        if problem and not self._paused_for(CONNECTIVITY_PAUSE_REASON):
            self._pause(CONNECTIVITY_PAUSE_REASON)
            self.notify(
                "\n".join(
                    [
                        f"CONNECTIVITY LOST: {problem}. New entries paused 1h; "
                        "the core and protective stops keep running.",
                        *error_lines(drain_alerts()),
                    ]
                )
            )
        return "ran"

    def _claim_cycle(self) -> tuple[str, bool]:
        """Return (cycle_id, resumed): the in-flight cycle to replay, or a fresh id. Never raises.

        A cycle whose replay was cut short too is abandoned rather than replayed forever; with Redis
        unreachable the fresh id is not recorded, so it cannot be resumed.
        """
        redis = self._deps.kill_switch.redis
        try:
            raw = redis.get(IN_FLIGHT_KEY)
            marker = raw.decode() if isinstance(raw, bytes) else raw
            if marker is not None and not marker.endswith(RESUMED_SUFFIX):
                redis.set(IN_FLIGHT_KEY, marker + RESUMED_SUFFIX, ex=IN_FLIGHT_TTL_SECONDS)
                return marker, True
            if marker is not None:
                log.error(
                    "cycle_abandoned",
                    cycle_id=marker.removesuffix(RESUMED_SUFFIX),
                    note="its replay was cut short too; not resumed again",
                )
            cycle_id = str(uuid4())
            redis.set(IN_FLIGHT_KEY, cycle_id, ex=IN_FLIGHT_TTL_SECONDS)
            return cycle_id, False
        except RedisError:
            log.warning("cycle_marker_unavailable", note="this cycle cannot be resumed")
            return str(uuid4()), False

    def _release_cycle(self) -> None:
        """Forget the in-flight cycle; a failed delete just leaves the marker to expire."""
        try:
            self._deps.kill_switch.redis.delete(IN_FLIGHT_KEY)
        except RedisError:
            log.warning("cycle_marker_unavailable", note="marker left to expire")

    def notify(self, text: str) -> None:
        """Post a scheduler-level notice to Discord, prefixed with the mode."""
        if self._deps.notifier is not None:
            self._deps.notifier.send(f"[{self._settings.trading_mode}] {text}")

    def _paused_for(self, reason: str) -> bool:
        """Whether this runner pause, rather than another halt, is the active halt."""
        return self._deps.kill_switch.is_halted()[1] == f"cooldown:{reason}"

    def _pause(self, reason: str) -> None:
        """Arm the one-hour breaker cooldown (never the kill switch)."""
        self._deps.kill_switch.start_cooldown(reason, DEFAULT_LIMITS)
        log.warning("cycles_paused", reason=reason, seconds=DEFAULT_LIMITS.breaker_cooldown_seconds)


def cycle_trigger(interval_minutes: int) -> CronTrigger:
    """A UTC cron trigger every `interval_minutes` on the clock; ValueError unless it divides 60."""
    if interval_minutes <= 0 or 60 % interval_minutes:
        raise ValueError(f"CYCLE_INTERVAL_MINUTES must divide 60, got {interval_minutes}")
    return CronTrigger(minute=f"*/{interval_minutes}", timezone="UTC")


async def serve(
    settings: Settings, deps: CycleDeps, graph: CompiledStateGraph, engine: Engine | None
) -> None:
    """Run the bot, the scheduler and a first cycle until cancelled.

    Raises ValueError on a malformed Discord setting and LoginFailure on a rejected token, both
    before the first cycle.
    """
    # Build the bot first: a bad Discord setting must fail before anything runs.
    bot = None
    if settings.discord_bot_token and settings.discord_channel_id and settings.discord_owner_id:
        bot = DiscordBot(
            settings.discord_bot_token,
            settings.discord_channel_id,
            settings.discord_owner_id,
            deps.kill_switch,
            deps.brokers,
            deps.tracker,
        )
    runner = AgentRunner(
        settings,
        deps,
        lambda stocks_open, cycle_id: run_cycle(graph, cycle_id, stock_market_open=stocks_open),
    )
    cleared = deps.kill_switch.clear_cooldown(*RUNNER_PAUSE_REASONS)
    if cleared:
        log.warning("runner_pause_cleared_on_restart", reason=cleared)

    async def prune() -> None:
        """Daily housekeeping job: drop graph checkpoints of old cycles (never raises)."""
        cutoff = datetime.now(UTC) - timedelta(days=PRUNE_AFTER_DAYS)
        try:
            await asyncio.to_thread(prune_checkpoints, engine, cutoff)
        except Exception:
            log.warning("checkpoint_prune_failed", exc_info=True)

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        runner.tick,
        cycle_trigger(settings.cycle_interval_minutes),
        id="cycle",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=MISFIRE_GRACE_SECONDS,
    )
    if engine is not None:
        scheduler.add_job(prune, CronTrigger(hour=3, minute=7, timezone="UTC"), id="prune")

    # discord.Client must be ENTERED (its async context manager initialises the
    # gateway state) before start()/wait_until_ready(); the exit closes it.
    async with bot if bot is not None else contextlib.nullcontext():
        bot_task = None
        if bot is not None:
            bot_task = asyncio.create_task(bot.start(settings.discord_bot_token))
            waited = 0
            while True:  # fail closed: no cycle runs until the kill switch is reachable
                try:
                    await asyncio.wait_for(
                        bot.wait_until_ready(), timeout=BOT_READY_TIMEOUT_SECONDS
                    )
                    break
                except TimeoutError:
                    if bot_task.done():
                        bot_task.result()  # surfaces LoginFailure / a DM channel id
                    waited += BOT_READY_TIMEOUT_SECONDS
                    log.warning(
                        "discord_gateway_not_ready",
                        waited_seconds=waited,
                        note="cycles wait for the kill switch; restart to give up",
                    )
        scheduler.start()
        runner.notify(
            f"agent started: every {settings.cycle_interval_minutes} min, stocks "
            f"{settings.stock_symbols or 'none'}, crypto {settings.crypto_symbols or 'none'}, "
            f"commands {'on' if bot else 'off'}"
        )
        log.info(
            "scheduler_started",
            interval_minutes=settings.cycle_interval_minutes,
            bot=bot is not None,
        )
        try:
            await runner.tick()  # a restart must not wait for the next slot
            if bot_task is not None:
                await bot_task
            else:
                await asyncio.Event().wait()
        finally:
            scheduler.shutdown(wait=False)
            runner.notify("agent stopping (a cycle in flight finishes first)")
            log.info("scheduler_stopped")
