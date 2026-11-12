"""Kill switch and breaker cooldowns, kept in Redis and checked everywhere.

- Kill switch: no TTL. Set by Discord /killswitch, by max drawdown, or by hand. Only a
  human clears it; nothing auto-resumes.
- Breaker cooldown: a TTL of at least an hour. The gate re-checks breakers every cycle,
  so trading resumes on expiry only if the condition has cleared.

Fail-closed: if Redis is unreachable, is_halted() returns True. Losing the ability to
check the safety flag must stop trading, not bypass it.
"""

from redis import Redis, RedisError

from trading_agent.log import get_logger
from trading_agent.risk.limits import RiskLimits

log = get_logger(__name__)

KILL_SWITCH_KEY = "risk:kill_switch"
COOLDOWN_KEY = "risk:breaker_cooldown"
# Set by the human /killswitch. The executor cancels open orders only for this reason:
# an automatic drawdown trip halts new entries but leaves stops and stop sales working.
MANUAL_KILL_SWITCH_REASON = "manual_discord"
# Set by the gate when drawdown trips; it stops new entries only.
AUTO_KILL_SWITCH_REASON = "max_drawdown_exceeded"
# The scheduler's own one-hour pauses, stored as breaker cooldowns.
CONNECTIVITY_PAUSE_REASON = "connectivity_lost"
CRASH_PAUSE_REASON = "cycle_crashed"
RUNNER_PAUSE_REASONS = (CONNECTIVITY_PAUSE_REASON, CRASH_PAUSE_REASON)


class KillSwitch:
    def __init__(self, redis: Redis) -> None:
        """Wrap a Redis client for halt-state reads and writes."""
        self._redis = redis

    @property
    def redis(self) -> Redis:
        """The Redis client (the scheduler keeps its in-flight cycle id there too)."""
        return self._redis

    def activate(self, reason: str) -> None:
        """Arm the no-TTL kill switch; never raises.

        If Redis is down the caller still rejects the batch and is_halted fails closed;
        the breaker retries the write next cycle, so the flag lands once Redis returns.
        """
        try:
            self._redis.set(KILL_SWITCH_KEY, reason)
            log.error("kill_switch_activated", reason=reason)
        except RedisError:
            log.exception("kill_switch_persist_failed", reason=reason)

    def deactivate_human_only(self, confirmed_by: str) -> None:
        """Clear the kill switch; only a human instruction path may call this.

        A RedisError is deliberately not swallowed: a failed clear must be visible.
        """
        self._redis.delete(KILL_SWITCH_KEY)
        log.warning("kill_switch_deactivated", confirmed_by=confirmed_by)

    def start_cooldown(
        self, reason: str, limits: RiskLimits, ttl_seconds: int | None = None
    ) -> None:
        """Arm a breaker cooldown, never under the 1-hour minimum; never raises."""
        ttl = max(limits.breaker_cooldown_seconds, ttl_seconds or 0)
        try:
            self._redis.set(COOLDOWN_KEY, reason, ex=ttl)
            # Error level: a breaker trip pages the human.
            log.error("breaker_cooldown_started", reason=reason, ttl=ttl)
        except RedisError:
            log.exception("breaker_cooldown_persist_failed", reason=reason)

    def clear_cooldown(self, *reasons: str) -> str | None:
        """Clear the cooldown only if its reason is one of `reasons`; return it, else None.

        The reason check lets the scheduler release its own pauses on a restart without
        touching a gate breaker's cooldown. Never raises.
        """
        try:
            current = self._redis.get(COOLDOWN_KEY)
            if current is None or _text(current) not in reasons:
                return None
            self._redis.delete(COOLDOWN_KEY)
            log.warning("breaker_cooldown_cleared", reason=_text(current))
            return _text(current)
        except RedisError:
            log.exception("breaker_cooldown_clear_failed")
            return None

    def is_halted(self) -> tuple[bool, str]:
        """(halted, reason), failing closed to (True, "redis_unreachable"); never raises.

        The reason is "kill_switch:<reason>" or "cooldown:<reason>", or "" when not halted.
        """
        try:
            kill = self._redis.get(KILL_SWITCH_KEY)
            if kill is not None:
                return True, f"kill_switch:{_text(kill)}"
            cooldown = self._redis.get(COOLDOWN_KEY)
            if cooldown is not None:
                return True, f"cooldown:{_text(cooldown)}"
            return False, ""
        except RedisError:
            log.exception("kill_switch_check_failed_failing_closed")
            return True, "redis_unreachable"


def stops_everything(halted: bool, halt_reason: str) -> bool:
    """The halt policy: whether this halt stops every order, or only the swing sleeve's.

    Any halt stops new swing entries and signal sells. Cooldowns and the automatic
    drawdown kill switch stop only those: stop sales, the core's orders and protective
    stops continue. True (the scheduler skips the cycle) for the manual kill switch,
    an unreachable Redis (a human may have armed the switch) and any unknown reason.
    """
    return halted and not (
        halt_reason.startswith("cooldown:")
        or halt_reason == f"kill_switch:{AUTO_KILL_SWITCH_REASON}"
    )


def scheduler_paused(halt_reason: str) -> bool:
    """Whether the active halt is the scheduler's own crash or connectivity pause."""
    return halt_reason in {f"cooldown:{r}" for r in RUNNER_PAUSE_REASONS}


def _text(value: bytes | str) -> str:
    """Normalize a Redis reply to str."""
    return value.decode() if isinstance(value, bytes) else str(value)
