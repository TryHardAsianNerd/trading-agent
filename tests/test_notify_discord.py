"""Discord notifier + command bot tests (D-017). No network.

The REST sender runs against an httpx MockTransport; the bot's command
handlers and its authorization check are invoked directly with a fake
Interaction (discord.py's gateway is never connected). Kill-switch state
uses the shared FakeRedis.
"""

import asyncio
import contextlib
import io
import json
import logging
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from redis import RedisError
from tests.fakes import BrokenRedis, FakeRedis

from trading_agent.brokers.base import AccountSummary, BrokerError, BrokerPosition
from trading_agent.config import Settings
from trading_agent.graph.nodes.notifier import notify
from trading_agent.log import (
    ALERT_BUFFER_MAX,
    collect_alerts,
    configure_logging,
    drain_alerts,
    get_logger,
)
from trading_agent.notify.discord import (
    MESSAGE_LIMIT,
    DiscordBot,
    DiscordNotifier,
    format_crash_alert,
    format_cycle_alert,
)
from trading_agent.risk.killswitch import COOLDOWN_KEY, KILL_SWITCH_KEY, KillSwitch
from trading_agent.risk.limits import RiskLimits

TOKEN = "test-token-not-real"
CHANNEL = "111"
OWNER = "222"


# ------------------------------------------------------------- REST send


def _notifier(handler):
    return DiscordNotifier(
        TOKEN, CHANNEL, client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_send_posts_to_the_channel_with_bot_auth():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"id": "1"})

    assert _notifier(handler).send("hello") is True
    assert seen["url"].endswith(f"/channels/{CHANNEL}/messages")
    assert seen["auth"] == f"Bot {TOKEN}"
    assert seen["body"] == {"content": "hello"}


def test_send_truncates_to_discord_limit():
    seen = {}

    def handler(request):
        seen["content"] = json.loads(request.read())["content"]
        return httpx.Response(200)

    _notifier(handler).send("x" * (MESSAGE_LIMIT + 500))
    assert len(seen["content"]) == MESSAGE_LIMIT


def test_send_failures_return_false_and_never_raise():
    assert _notifier(lambda r: httpx.Response(500)).send("x") is False

    def boom(request):
        raise httpx.ConnectError("down")

    assert _notifier(boom).send("x") is False


def test_notifier_requires_credentials():
    with pytest.raises(ValueError):
        DiscordNotifier("", CHANNEL)
    with pytest.raises(ValueError):
        DiscordNotifier(TOKEN, "")


# ---------------------------------------------------------- alert format


def test_quiet_cycle_produces_no_alert():
    state = {"cycle_id": "abc", "proposals": [], "rejections": [{"reason": "insufficient_data"}]}
    assert format_cycle_alert(state, [], "paper") is None


def test_alert_puts_errors_first_then_halt_results_and_breaches():
    state = {
        "cycle_id": "abcdef12-rest",
        "execution_results": [
            {
                "kind": "order",
                "side": "buy",
                "quantity": "0.5",
                "symbol": "BTC-USD",
                "status": "filled",
                "broker_order_id": "529",
            },
            {
                "kind": "stop_breach",
                "side": "sell",
                "quantity": "5",
                "symbol": "T-X",
                "status": "failed",
                "detail": "venue down https://venue/secret",
            },
        ],
        "stop_breaches": [{"symbol": "T-X", "current_price": "90", "stop_price": "94"}],
    }
    # `error`/`detail` carry raw SDK text and never render (D-018).
    errors = [
        {
            "event": "position_unprotected",
            "level": "error",
            "symbol": "ETH-USD",
            "error": "503 https://venue/secret",
        }
    ]
    text = format_cycle_alert(state, errors, "paper")
    assert text.splitlines() == [
        "[paper] cycle abcdef12",
        "ERROR position_unprotected symbol=ETH-USD",
        "BUY 0.5 BTC-USD (order) -> filled #529",
        "SELL 5 T-X (stop sale) -> failed",
        "Stop breach T-X @ 90 (stop 94)",
    ]
    assert "secret" not in text


def test_alert_shows_halt_once_and_not_per_refused_submission():
    state = {
        "cycle_id": "c",
        "execution_results": [
            {
                "kind": "order",
                "side": "buy",
                "quantity": "1",
                "symbol": "A",
                "status": "halted",
                "detail": "kill_switch:manual_discord",
            },
            {
                "kind": "stop_breach",
                "side": "sell",
                "quantity": "2",
                "symbol": "B",
                "status": "halted",
                "detail": "kill_switch:manual_discord",
            },
        ],
    }
    lines = format_cycle_alert(state, [], "paper").splitlines()
    assert lines[1] == "HALTED: kill_switch:manual_discord (2 submission(s) refused)"
    assert lines[2] == "BUY 1 A (order) -> halted"


def test_alert_is_truncated_within_limit_keeping_errors():
    state = {
        "cycle_id": "c",
        "execution_results": [
            {"kind": "order", "side": "buy", "quantity": "1", "symbol": f"S{i}", "status": "filled"}
            for i in range(200)
        ],
    }
    errors = [{"event": "order_lost_at_venue", "level": "error", "symbol": "LATE"}]
    text = format_cycle_alert(state, errors, "paper")
    assert len(text) <= MESSAGE_LIMIT and text.endswith("(truncated)")
    assert "ERROR order_lost_at_venue symbol=LATE" in text  # errors survive the cut


def test_crash_alert_carries_type_only_plus_buffered_errors():
    errors = [{"event": "position_unprotected", "level": "error", "symbol": "BTC-USD"}]
    text = format_crash_alert(RuntimeError("postgresql://user:secret@host"), errors, "paper")
    assert text.splitlines() == [
        "[paper] CYCLE CRASHED: RuntimeError",
        "ERROR position_unprotected symbol=BTC-USD",
    ]
    assert "secret" not in text


# ---------------------------------------------------------- log capture


def test_collect_alerts_keeps_our_error_level_scalar_fields_only():
    drain_alerts()
    collect_alerts(None, "info", {"event": "fine", "level": "info"})
    collect_alerts(
        None, "error", {"event": "bad", "level": "error", "symbol": "X", "exc_info": object()}
    )
    # Third-party (stdlib) error records carry uncontrolled text: not collected.
    collect_alerts(
        None, "error", {"event": "lib blew up http://x", "level": "error", "_from_structlog": False}
    )
    assert drain_alerts() == [{"event": "bad", "level": "error", "symbol": "X"}]
    assert drain_alerts() == []  # drained


def test_configured_pipeline_collects_our_errors_and_not_stdlib_records():
    """End to end through configure_logging: the structlog event is collected
    with its level and fields; a third-party stdlib record never is."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    drain_alerts()
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # the JSON stream, not test output
            configure_logging()
            get_logger("trading_agent.tests.pipeline").error("event_x", symbol="X")
            logging.getLogger("some.lib").error("lib text")
        items = drain_alerts()
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
    assert len(items) == 1
    assert items[0]["event"] == "event_x"
    assert items[0]["level"] == "error"
    assert items[0]["symbol"] == "X"


def test_collect_alerts_is_capped_and_reports_the_overflow():
    drain_alerts()
    for i in range(ALERT_BUFFER_MAX + 10):
        collect_alerts(None, "error", {"event": f"e{i}", "level": "error"})
    items = drain_alerts()
    assert len(items) == ALERT_BUFFER_MAX + 1
    assert items[-1] == {"event": "alerts_dropped", "level": "error", "count": 10}
    assert drain_alerts() == []


# --------------------------------------------------------- notifier node


class RecordingNotifier:
    def __init__(self, ok=True):
        self.sent = []
        self.ok = ok

    def send(self, text):
        self.sent.append(text)
        return self.ok


def _deps(notifier, redis=None):
    return SimpleNamespace(
        notifier=notifier,
        settings=Settings(_env_file=None),
        kill_switch=KillSwitch(redis or FakeRedis()),
    )


def test_notifier_node_sends_only_when_something_happened():
    drain_alerts()
    rec = RecordingNotifier()
    assert notify({"cycle_id": "c"}, deps=_deps(rec)) == {"notifications_sent": ["log"]}
    assert rec.sent == []
    state = {
        "cycle_id": "c",
        "execution_results": [
            {"kind": "order", "side": "buy", "quantity": "1", "symbol": "X", "status": "filled"}
        ],
    }
    assert notify(state, deps=_deps(rec)) == {"notifications_sent": ["log", "discord"]}
    assert "BUY 1 X" in rec.sent[0]


def test_notifier_node_drains_errors_even_without_a_channel():
    drain_alerts()
    collect_alerts(None, "error", {"event": "bad", "level": "error"})
    assert notify({"cycle_id": "c"}, deps=_deps(None)) == {"notifications_sent": ["log"]}
    assert drain_alerts() == []


def test_during_a_scheduler_pause_only_new_kinds_of_error_page(monkeypatch):
    """The cycle that started the pause paged its errors; the pause's cycles page a kind of
    error only the first time, and their fills still post."""
    from trading_agent.graph.nodes import notifier as node

    monkeypatch.setattr(node, "_PAGED_KINDS", set())
    rec, redis = RecordingNotifier(), FakeRedis()

    def cycle(event, state=None):
        drain_alerts()
        collect_alerts(None, "error", {"event": event, "level": "error"})
        notify(state or {"cycle_id": "c"}, deps=_deps(rec, redis))

    cycle("order_lookup_failed")  # before the pause: pages and starts the record
    KillSwitch(redis).start_cooldown("connectivity_lost", RiskLimits())
    cycle("order_lookup_failed")  # a repeat during the pause: held
    fill = {"kind": "order", "side": "sell", "quantity": "1", "symbol": "VOO", "status": "filled"}
    cycle("order_lookup_failed", {"cycle_id": "c", "execution_results": [fill]})
    cycle("account_drawdown_alert")  # new during the pause: pages at once
    assert len(rec.sent) == 3
    assert "order_lookup_failed" in rec.sent[0]
    assert "SELL 1 VOO" in rec.sent[1] and "order_lookup_failed" not in rec.sent[1]
    assert "account_drawdown_alert" in rec.sent[2]


def test_under_a_breaker_cooldown_errors_still_page():
    redis = FakeRedis()
    KillSwitch(redis).start_cooldown("daily_loss_breaker", RiskLimits())
    drain_alerts()
    collect_alerts(None, "error", {"event": "position_unprotected", "level": "error"})
    rec = RecordingNotifier()
    notify({"cycle_id": "c"}, deps=_deps(rec, redis))
    assert "position_unprotected" in rec.sent[0]


def test_notifier_node_survives_a_failed_send():
    drain_alerts()
    state = {"cycle_id": "c", "stop_breaches": [{"symbol": "X"}]}
    assert notify(state, deps=_deps(RecordingNotifier(ok=False))) == {"notifications_sent": ["log"]}


# ------------------------------------------------------------------ bot


class FakeInteraction:
    def __init__(self, user_id, channel_id=int(CHANNEL), ack_fails=False):
        self.user = SimpleNamespace(id=user_id)
        self.channel_id = channel_id
        self.command = SimpleNamespace(name="status")
        self.messages = []
        self.deferred = False
        self._ack_fails = ack_fails
        self.response = SimpleNamespace(defer=self._defer)
        self.followup = SimpleNamespace(send=self._send)

    async def _send(self, text, **kwargs):
        self.messages.append(text)

    async def _defer(self, **kwargs):
        if self._ack_fails:
            raise RuntimeError("interaction expired")
        self.deferred = True


class FakeVenue:
    def __init__(self, name, cancelled=3, fail=False, positions=()):
        self.name = name
        self.asset_class = "crypto"
        self._cancelled = cancelled
        self._fail = fail
        self._positions = list(positions)
        self.cancel_calls = 0

    def cancel_all_orders(self):
        self.cancel_calls += 1
        if self._fail:
            raise BrokerError("venue down")
        return self._cancelled

    def get_positions(self):
        return self._positions

    def get_account(self):
        return AccountSummary(equity=Decimal("10500"), cash=Decimal("9000"))


class FakeTracker:
    def __init__(self, brokers):
        self._brokers = brokers

    def build_state(self, prices, now=None):
        from trading_agent.risk.models import PortfolioState, Position

        positions = {}
        for b in self._brokers:
            for p in b.get_positions():
                positions[p.symbol] = Position(
                    symbol=p.symbol,
                    asset_class=p.asset_class,
                    quantity=p.quantity,
                    avg_entry_price=p.avg_entry_price,
                )
        return PortfolioState(
            equity=Decimal("10500"),
            cash=Decimal("9000"),
            peak_equity=Decimal("10500"),
            day_start_equity=Decimal("10500"),
            week_start_equity=Decimal("10500"),
            consecutive_losses=0,
            positions=positions,
        )


class WriteOnlyBrokenRedis(FakeRedis):
    """Reads work, writes fail: the OOM / read-only-replica shape."""

    def set(self, key, value, ex=None):
        raise RedisError("READONLY You can't write against a read only replica")


def _bot(redis=None, brokers=None):
    brokers = brokers if brokers is not None else [FakeVenue("paper")]
    return DiscordBot(
        TOKEN, CHANNEL, OWNER, KillSwitch(redis or FakeRedis()), brokers, FakeTracker(brokers)
    )


def run(coro):
    return asyncio.run(coro)


def test_bot_requires_all_three_credentials():
    with pytest.raises(ValueError):
        DiscordBot(TOKEN, CHANNEL, "", KillSwitch(FakeRedis()), [], FakeTracker([]))


def test_commands_are_bound_to_the_bot_and_gated_by_one_check():
    bot = _bot()
    assert {c.name for c in bot.tree.get_commands()} == {"killswitch", "resume", "status"}
    assert bot.tree.get_command("killswitch").binding is bot
    assert run(bot.tree.interaction_check(FakeInteraction(user_id=int(OWNER)))) is True
    assert run(bot.tree.interaction_check(FakeInteraction(user_id=999))) is False
    assert (
        run(bot.tree.interaction_check(FakeInteraction(user_id=int(OWNER), channel_id=5))) is False
    )


def test_killswitch_arms_first_then_cancels_everywhere_and_reports():
    redis = FakeRedis()
    venues = [FakeVenue("paper", cancelled=2), FakeVenue("alpaca", fail=True)]
    bot = _bot(redis, venues)
    owner = FakeInteraction(user_id=int(OWNER))

    run(bot.killswitch(owner))

    assert redis.store[KILL_SWITCH_KEY] == "manual_discord"
    assert KILL_SWITCH_KEY not in redis.ttls  # durable, no expiry
    assert all(v.cancel_calls == 1 for v in venues)
    (reply,) = owner.messages
    assert reply.startswith("KILL SWITCH ARMED. ")
    assert "paper: cancelled 2" in reply and "alpaca: CANCEL FAILED (BrokerError)" in reply
    assert reply.endswith("Re-run /killswitch if a cycle was in flight.")


def test_killswitch_warns_when_the_flag_did_not_persist():
    """activate() swallows Redis write errors; a human command has no retry,
    so a write that did not land must be shouted, never reported as armed."""
    bot = _bot(WriteOnlyBrokenRedis())
    owner = FakeInteraction(user_id=int(OWNER))
    run(bot.killswitch(owner))
    (reply,) = owner.messages
    assert reply.startswith("WARNING: KILL-SWITCH FLAG NOT PERSISTED")
    assert "paper: cancelled 3" in reply  # cancels still ran


def test_killswitch_arms_even_if_discord_ack_fails():
    redis = FakeRedis()
    bot = _bot(redis)
    owner = FakeInteraction(user_id=int(OWNER), ack_fails=True)
    run(bot.killswitch(owner))
    assert redis.store[KILL_SWITCH_KEY] == "manual_discord"
    assert owner.messages[0].startswith("KILL SWITCH ARMED")  # follow-up still attempted


def test_killswitch_with_fully_broken_redis_still_cancels_and_warns():
    bot = _bot(BrokenRedis())
    owner = FakeInteraction(user_id=int(OWNER))
    run(bot.killswitch(owner))
    assert owner.messages[0].startswith("WARNING: KILL-SWITCH FLAG NOT PERSISTED")


def test_resume_clears_the_flag_through_the_human_only_path_and_echoes_the_reason():
    redis = FakeRedis()
    bot = _bot(redis)
    KillSwitch(redis).activate("max_drawdown")
    owner = FakeInteraction(user_id=int(OWNER))

    run(bot.resume(owner))

    assert KILL_SWITCH_KEY not in redis.store
    assert owner.deferred
    assert owner.messages == ["Cleared kill_switch:max_drawdown. Trading resumes next cycle."]


def test_resume_with_nothing_set_clears_nothing():
    redis = FakeRedis()
    owner = FakeInteraction(user_id=int(OWNER))
    run(_bot(redis).resume(owner))
    assert owner.messages == ["Nothing to clear: trading is enabled."]


def test_resume_during_a_cooldown_does_not_touch_redis():
    class CountingRedis(FakeRedis):
        deletes = 0

        def delete(self, key):
            self.deletes += 1
            super().delete(key)

    redis = CountingRedis()
    KillSwitch(redis).start_cooldown("daily_loss", RiskLimits())
    owner = FakeInteraction(user_id=int(OWNER))
    run(_bot(redis).resume(owner))
    assert redis.deletes == 0 and COOLDOWN_KEY in redis.store
    assert owner.messages == ["No kill switch to clear: halted by cooldown:daily_loss."]


def test_resume_with_unreachable_redis_reports_the_fail_closed_halt():
    owner = FakeInteraction(user_id=int(OWNER))
    run(_bot(BrokenRedis()).resume(owner))
    assert owner.messages == ["No kill switch to clear: halted by redis_unreachable."]


def test_resume_reports_a_failed_clear_by_type_only():
    class DeleteBrokenRedis(FakeRedis):
        def delete(self, key):
            raise RedisError("READONLY You can't write against a read only replica")

    redis = DeleteBrokenRedis()
    KillSwitch(redis).activate("manual_discord")
    owner = FakeInteraction(user_id=int(OWNER))
    run(_bot(redis).resume(owner))
    assert owner.messages == ["Resume FAILED: RedisError"]
    assert KILL_SWITCH_KEY in redis.store  # still armed, and the human knows


def test_status_reports_halt_state_equity_and_positions():
    redis = FakeRedis()
    position = BrokerPosition(
        symbol="BTC-USD",
        asset_class="crypto",
        quantity=Decimal("0.0066000000"),
        avg_entry_price=Decimal("75986.67"),
    )
    bot = _bot(redis, [FakeVenue("paper", positions=[position])])
    owner = FakeInteraction(user_id=int(OWNER))

    run(bot.status(owner))
    (reply,) = owner.messages
    assert reply.splitlines() == [
        "Trading: enabled",
        "Equity 10500.00  cash 9000.00",
        "BTC-USD: 0.0066 @ 75986.67",
    ]

    KillSwitch(redis).activate("manual_discord")
    owner = FakeInteraction(user_id=int(OWNER))
    run(bot.status(owner))
    assert owner.messages[0].startswith("Halted: kill_switch:manual_discord")


def test_status_survives_a_portfolio_failure():
    class DownTracker:
        def build_state(self, prices, now=None):
            raise RuntimeError("db down")

    bot = DiscordBot(TOKEN, CHANNEL, OWNER, KillSwitch(FakeRedis()), [], DownTracker())
    owner = FakeInteraction(user_id=int(OWNER))
    run(bot.status(owner))
    assert owner.messages == ["Trading: enabled\nPortfolio unavailable: RuntimeError"]


def test_bot_rejects_non_integer_ids_with_a_clear_message():
    with pytest.raises(ValueError, match="plain integer ids"):
        DiscordBot(
            TOKEN, CHANNEL, "123456789012345678Disc", KillSwitch(FakeRedis()), [], FakeTracker([])
        )
    with pytest.raises(ValueError, match="plain integer ids"):
        DiscordBot(TOKEN, "abc", OWNER, KillSwitch(FakeRedis()), [], FakeTracker([]))


def test_alert_names_a_reprotect_stop():
    """A stop re-placed for an unprotected position is called out as such."""
    state = {
        "cycle_id": "abcdef1234",
        "execution_results": [
            {
                "kind": "reprotect",
                "side": "sell",
                "quantity": "5",
                "symbol": "AAPL",
                "status": "placed",
                "stop_price": "94",
            }
        ],
    }
    text = format_cycle_alert(state, [], "paper")
    assert "SELL 5 AAPL (re-protect stop) -> placed" in text.splitlines()
