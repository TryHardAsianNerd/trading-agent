"""Discord: cycle alerts over REST and the owner-only command bot over the gateway.

Send failures are warnings, not errors, so a failing channel cannot feed the next alert.
No exception text reaches a message or a log line here, only type names and ids: venue SDK
text can carry secrets, and the full text stays in the JSON logs.
"""

from __future__ import annotations

import asyncio

import discord
import httpx
from discord import app_commands

from trading_agent.brokers.base import Broker
from trading_agent.graph.state import AgentState
from trading_agent.log import get_logger
from trading_agent.portfolio.tracker import PortfolioTracker
from trading_agent.risk.killswitch import MANUAL_KILL_SWITCH_REASON, KillSwitch

log = get_logger(__name__)

DISCORD_API = "https://discord.com/api/v10"
MESSAGE_LIMIT = 2000
# In display order. Never `error` or `detail`: those carry raw exception text.
_ALERT_FIELDS = (
    "symbol",
    "broker",
    "order_id",
    "broker_order_id",
    "reason",
    "count",
)
_TRUNCATED = "\n... (truncated)"


class DiscordNotifier:
    """Posts alert messages to one channel via the REST API (no gateway)."""

    def __init__(self, token: str, channel_id: str, client: httpx.Client | None = None) -> None:
        """Build the sender; ValueError if the token or channel id is empty."""
        if not token or not channel_id.strip():
            raise ValueError("DiscordNotifier requires DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID")
        self._url = f"{DISCORD_API}/channels/{channel_id.strip()}/messages"
        self._headers = {"Authorization": f"Bot {token}"}
        self._client = client or httpx.Client(timeout=10.0)

    def send(self, text: str) -> bool:
        """Post one message cut to Discord's limit; True when Discord accepted it. Never raises."""
        try:
            response = self._client.post(
                self._url, headers=self._headers, json={"content": text[:MESSAGE_LIMIT]}
            )
        except Exception as e:
            log.warning("discord_send_failed", reason=type(e).__name__)
            return False
        if response.status_code >= 300:
            log.warning("discord_send_failed", status=response.status_code)
            return False
        return True


def error_lines(errors: list[dict]) -> list[str]:
    """Render collected error-level log events as "ERROR <event> k=v ..." lines."""
    lines = []
    for e in errors:
        fields = " ".join(f"{k}={e[k]}" for k in _ALERT_FIELDS if k in e)
        lines.append(f"ERROR {e.get('event')} {fields}".rstrip())
    return lines


def format_cycle_alert(state: AgentState, errors: list[dict], trading_mode: str) -> str | None:
    """Render one cycle's alert, or None when nothing happened worth a ping.

    Errors come first because truncation cuts the tail. Routine rejections are left out: they
    would page every cycle.
    """
    lines = error_lines(errors)
    results = state.get("execution_results") or []
    halted = [r for r in results if r.get("status") == "halted"]
    if halted:
        lines.append(f"HALTED: {halted[0].get('detail')} ({len(halted)} submission(s) refused)")
    for r in results:
        venue = f" #{r['broker_order_id']}" if r.get("broker_order_id") else ""
        kind = {"stop_breach": "stop sale", "reprotect": "re-protect stop"}.get(
            str(r.get("kind")), "order"
        )
        lines.append(
            f"{str(r.get('side', '')).upper()} {r.get('quantity')} {r.get('symbol')} "
            f"({kind}) -> {r.get('status')}{venue}"
        )
    for b in state.get("stop_breaches") or []:
        lines.append(
            f"Stop breach {b.get('symbol')} @ {b.get('current_price')} (stop {b.get('stop_price')})"
        )
    if not lines:
        return None
    cycle = str(state.get("cycle_id", ""))[:8]
    return _clip("\n".join([f"[{trading_mode}] cycle {cycle}", *lines]))


def format_crash_alert(exc: BaseException, errors: list[dict], trading_mode: str) -> str:
    """The notice for a cycle that died before the notifier ran, with the errors buffered so far."""
    header = f"[{trading_mode}] CYCLE CRASHED: {type(exc).__name__}"
    return _clip("\n".join([header, *error_lines(errors)]))


def _clip(text: str) -> str:
    """Cut a message to Discord's limit with a visible marker."""
    if len(text) <= MESSAGE_LIMIT:
        return text
    return text[: MESSAGE_LIMIT - len(_TRUNCATED)] + _TRUNCATED


class _OwnerOnlyTree(app_commands.CommandTree):
    """Command tree whose single check rejects everyone but the owner in one channel."""

    def __init__(self, client: discord.Client, owner_id: int, channel_id: int) -> None:
        """Bind the one allowed user and channel."""
        super().__init__(client)
        self._owner_id = owner_id
        self._channel_id = channel_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Allow only the owner in the configured channel; others are logged, never answered."""
        ok = interaction.user.id == self._owner_id and interaction.channel_id == self._channel_id
        command = getattr(interaction.command, "name", None)
        if ok:
            log.info("discord_command", command=command, user_id=interaction.user.id)
        else:
            log.warning(
                "discord_command_rejected",
                command=command,
                user_id=interaction.user.id,
                channel_id=interaction.channel_id,
            )
        return ok


class DiscordBot(discord.Client):
    """Owner-only slash commands over a gateway connection."""

    def __init__(
        self,
        token: str,
        channel_id: str,
        owner_id: str,
        kill_switch: KillSwitch,
        brokers: list[Broker],
        tracker: PortfolioTracker,
    ) -> None:
        """Wire the bot and its commands; ValueError on a missing setting or a non-integer id."""
        if not token or not channel_id or not owner_id:
            raise ValueError(
                "DiscordBot requires DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID and DISCORD_OWNER_ID"
            )
        if not (channel_id.strip().isdigit() and owner_id.strip().isdigit()):
            raise ValueError(
                "DISCORD_CHANNEL_ID and DISCORD_OWNER_ID must be plain integer ids "
                "(right-click -> Copy ID in Discord with Developer Mode on)"
            )
        super().__init__(intents=discord.Intents.none())
        self._token = token
        self._channel_id = int(channel_id)
        self._kill_switch = kill_switch
        self._brokers = brokers
        self._tracker = tracker
        self.tree = _OwnerOnlyTree(self, int(owner_id.strip()), self._channel_id)
        self.tree.command(name="killswitch", description="HALT trading and cancel all open orders")(
            self.killswitch
        )
        self.tree.command(name="resume", description="Clear the kill switch (human-only)")(
            self.resume
        )
        self.tree.command(name="status", description="Positions, equity, halt state")(self.status)

    async def setup_hook(self) -> None:
        """Register the commands on the channel's guild (syncs at once); ValueError on a DM."""
        channel = await self.fetch_channel(self._channel_id)
        guild = getattr(channel, "guild", None)
        if guild is None:
            raise ValueError("DISCORD_CHANNEL_ID must be a text channel in a server, not a DM")
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        log.info("discord_commands_synced", guild_id=guild.id)

    async def killswitch(self, interaction: discord.Interaction) -> None:
        """/killswitch: arm the kill switch, verify it persisted, then cancel everywhere.

        The flag is armed before Discord is acknowledged, so safety never waits on Discord, and read
        back, so a failed Redis write is never reported as armed.
        """
        armed = await asyncio.to_thread(self._arm)
        await self._ack(interaction)
        report = await asyncio.to_thread(self._cancel_everywhere)
        text = (
            (
                "KILL SWITCH ARMED. "
                if armed
                else "WARNING: KILL-SWITCH FLAG NOT PERSISTED "
                "(Redis write failed) - the agent halts only while Redis stays unreachable; "
                "retry /killswitch. "
            )
            + report
            + " Re-run /killswitch if a cycle was in flight."
        )
        log.warning("killswitch_command", armed=armed, report=report)
        await self._reply(interaction, text)

    async def resume(self, interaction: discord.Interaction) -> None:
        """/resume: clear the kill switch (the human-only path) and echo what was cleared."""
        await self._ack(interaction)
        confirmed_by = f"discord:{interaction.user.id}"
        try:
            text = await asyncio.to_thread(self._clear, confirmed_by)
        except Exception as e:  # a failed clear must be visible to the human
            text = f"Resume FAILED: {type(e).__name__}"
        await self._reply(interaction, text)

    async def status(self, interaction: discord.Interaction) -> None:
        """/status: halt state, equity, cash, and open positions."""
        await self._ack(interaction)
        await self._reply(interaction, await asyncio.to_thread(self._status_text))

    async def _ack(self, interaction: discord.Interaction) -> None:
        """Defer the interaction (3s ack window); a failed defer never aborts the command."""
        try:
            await interaction.response.defer()
        except Exception as e:
            log.warning("discord_ack_failed", reason=type(e).__name__)

    async def _reply(self, interaction: discord.Interaction, text: str) -> None:
        """Send the reply as a follow-up, best-effort: the command's effect already happened."""
        try:
            await interaction.followup.send(text[:MESSAGE_LIMIT])
        except Exception as e:
            log.warning("discord_reply_failed", reason=type(e).__name__)

    def _arm(self) -> bool:
        """Arm the kill switch and read it back; False when the write did not persist."""
        self._kill_switch.activate(MANUAL_KILL_SWITCH_REASON)
        halted, reason = self._kill_switch.is_halted()
        return halted and reason.startswith("kill_switch:")

    def _clear(self, confirmed_by: str) -> str:
        """Clear the kill switch only when it is what halts trading; the reply says what happened.

        A breaker cooldown or a `redis_unreachable` halt is not cleared by deleting a key, so the
        reply names what actually halts trading. A failed delete raises RedisError.
        """
        halted, reason = self._kill_switch.is_halted()
        if not halted:
            return "Nothing to clear: trading is enabled."
        if not reason.startswith("kill_switch:"):
            return f"No kill switch to clear: halted by {reason}."
        self._kill_switch.deactivate_human_only(confirmed_by)
        return f"Cleared {reason}. Trading resumes next cycle."

    def _cancel_everywhere(self) -> str:
        """Cancel open orders at every venue; report the count or failure type per venue."""
        parts = []
        for broker in self._brokers:
            try:
                parts.append(f"{broker.name}: cancelled {broker.cancel_all_orders()}")
            except Exception as e:
                log.exception("killswitch_cancel_failed", broker=broker.name)
                parts.append(f"{broker.name}: CANCEL FAILED ({type(e).__name__}) - check the venue")
        return "; ".join(parts) if parts else "no venues configured"

    def _status_text(self) -> str:
        """Assemble the /status reply from the kill switch and the tracker."""
        halted, reason = self._kill_switch.is_halted()
        lines = [f"Halted: {reason}" if halted else "Trading: enabled"]
        try:
            portfolio = self._tracker.build_state({})
        except Exception as e:
            lines.append(f"Portfolio unavailable: {type(e).__name__}")
            return "\n".join(lines)
        lines.append(f"Equity {portfolio.equity:.2f}  cash {portfolio.cash:.2f}")
        for symbol, p in sorted(portfolio.positions.items()):
            lines.append(f"{symbol}: {p.quantity.normalize():f} @ {p.avg_entry_price:.2f}")
        if not portfolio.positions:
            lines.append("No open positions")
        return "\n".join(lines)
