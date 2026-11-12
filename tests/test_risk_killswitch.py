"""Kill switch tests against a fake Redis, including the fail-closed contract."""

import pytest

from fakes import BrokenRedis, FakeRedis
from trading_agent.risk.killswitch import (
    COOLDOWN_KEY,
    KILL_SWITCH_KEY,
    KillSwitch,
    stops_everything,
)
from trading_agent.risk.limits import RiskLimits

LIMITS = RiskLimits()


def test_inactive_by_default():
    ks = KillSwitch(FakeRedis())
    assert ks.is_halted() == (False, "")


def test_activate_halts_with_reason():
    ks = KillSwitch(FakeRedis())
    ks.activate("max_drawdown_exceeded")
    halted, reason = ks.is_halted()
    assert halted
    assert reason == "kill_switch:max_drawdown_exceeded"


def test_kill_switch_has_no_ttl():
    """Nothing auto-resumes from a kill switch — no expiry may be set."""
    fake = FakeRedis()
    KillSwitch(fake).activate("manual")
    assert KILL_SWITCH_KEY not in fake.ttls


def test_deactivation_is_explicit():
    fake = FakeRedis()
    ks = KillSwitch(fake)
    ks.activate("manual")
    ks.deactivate_human_only(confirmed_by="jason-discord")
    assert ks.is_halted() == (False, "")


def test_cooldown_halts_with_ttl_of_at_least_an_hour():
    fake = FakeRedis()
    ks = KillSwitch(fake)
    ks.start_cooldown("daily_loss_breaker", LIMITS)
    halted, reason = ks.is_halted()
    assert halted
    assert reason == "cooldown:daily_loss_breaker"
    assert fake.ttls[COOLDOWN_KEY] >= 3600


def test_kill_switch_outranks_cooldown_in_reporting():
    fake = FakeRedis()
    ks = KillSwitch(fake)
    ks.start_cooldown("daily_loss_breaker", LIMITS)
    ks.activate("manual")
    assert ks.is_halted()[1].startswith("kill_switch:")


def test_redis_failure_fails_closed():
    """Losing Redis must halt trading, never bypass the check."""
    halted, reason = KillSwitch(BrokenRedis()).is_halted()
    assert halted
    assert reason == "redis_unreachable"


def test_bytes_values_decoded():
    fake = FakeRedis()
    fake.store[KILL_SWITCH_KEY] = b"manual"
    assert KillSwitch(fake).is_halted() == (True, "kill_switch:manual")


def test_clear_cooldown_only_clears_listed_reasons():
    fake = FakeRedis()
    ks = KillSwitch(fake)
    assert ks.clear_cooldown("cycle_crashed") is None  # nothing set
    ks.start_cooldown("daily_loss_breaker", LIMITS)
    assert ks.clear_cooldown("cycle_crashed", "connectivity_lost") is None  # not ours: untouched
    assert ks.is_halted() == (True, "cooldown:daily_loss_breaker")
    fake.delete(COOLDOWN_KEY)
    ks.start_cooldown("cycle_crashed", LIMITS)
    assert ks.clear_cooldown("cycle_crashed", "connectivity_lost") == "cycle_crashed"
    assert ks.is_halted() == (False, "")


def test_clear_cooldown_never_raises_on_redis_errors():
    assert KillSwitch(BrokenRedis()).clear_cooldown("cycle_crashed") is None


@pytest.mark.parametrize(
    ("halted", "reason", "everything"),
    [
        (False, "", False),
        (True, "cooldown:daily_loss_breaker", False),
        (True, "cooldown:cycle_crashed", False),
        (True, "cooldown:connectivity_lost", False),
        (True, "kill_switch:max_drawdown_exceeded", False),  # stops keep working
        (True, "kill_switch:manual_discord", True),  # cancel means cancel
        (True, "redis_unreachable", True),  # a human may have armed it: fail closed
        (True, "kill_switch:something_new", True),  # unknown reasons stop everything
    ],
)
def test_only_the_manual_switch_and_unknown_halts_stop_everything(halted, reason, everything):
    assert stops_everything(halted, reason) is everything
