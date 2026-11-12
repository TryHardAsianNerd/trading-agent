"""CycleDeps: the live resources every graph node needs, wired once at startup.

They travel here rather than in the state, which must stay serializable for the
checkpointer. Tests build a CycleDeps with fakes; production uses make_deps.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from redis import Redis
from sqlalchemy.orm import Session, sessionmaker

from trading_agent.brokers.alpaca import AlpacaBroker
from trading_agent.brokers.base import Broker
from trading_agent.brokers.paper import PaperBroker
from trading_agent.config import Settings
from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.data.providers.coingecko import CoinGeckoClient
from trading_agent.data.providers.fear_greed import FearGreedClient
from trading_agent.data.providers.finnhub import FinnhubClient
from trading_agent.data.providers.fred import FredClient
from trading_agent.graph.state import AgentState, TradeProposal
from trading_agent.log import get_logger
from trading_agent.ml.model import MODEL_DIR
from trading_agent.notify.discord import DiscordNotifier
from trading_agent.portfolio.tracker import PortfolioTracker
from trading_agent.risk.killswitch import KillSwitch

log = get_logger(__name__)


@dataclass
class CycleDeps:
    """Live resources shared by all graph nodes for one process lifetime.

    An optional client is None when its keys are not configured: no finnhub skips
    stock quotes, no alpaca_data puts bars and crypto quotes on Finnhub and CoinGecko,
    no fred leaves VIX unknown (no halving), and no notifier sends alerts to logs only.
    """

    settings: Settings
    session_factory: sessionmaker[Session]
    brokers: list[Broker]
    kill_switch: KillSwitch
    tracker: PortfolioTracker
    analyst: Callable[[AgentState], list[TradeProposal]]
    coingecko: CoinGeckoClient
    finnhub: FinnhubClient | None = None
    alpaca_data: AlpacaDataClient | None = None
    fred: FredClient | None = None
    notifier: DiscordNotifier | None = None
    fear_greed: FearGreedClient | None = None
    model_dir: Path = MODEL_DIR


def make_deps(settings: Settings, session_factory: sessionmaker[Session]) -> CycleDeps:
    """Wire production dependencies, skipping (with a warning) whatever has no keys.

    The analyst is Claude when ANTHROPIC_API_KEY is set, else the holds-only stub.
    """
    # Local import: llm_analyst type-hints CycleDeps, so importing at module level is circular.
    from trading_agent.graph.nodes.llm_analyst import ClaudeAnalyst, stub_analyst

    brokers: list[Broker] = []
    alpaca_data = None
    if settings.alpaca_api_key and settings.alpaca_secret_key:
        brokers.append(AlpacaBroker(settings))
        alpaca_data = AlpacaDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    else:
        log.warning("broker_unconfigured", venue="alpaca", note="no keys — stocks not tradable")

    coingecko = CoinGeckoClient()
    if settings.is_paper:
        # Crypto paper trades against the internal sim (D-013), filling at live quotes.
        quotes = alpaca_data or coingecko
        brokers.append(PaperBroker(settings, session_factory, lambda s: quotes.get_quote(s).price))
    else:
        log.warning("live_crypto_broker_not_wired", note="Phase 4 wiring lands with #27")

    # Short socket timeouts: /killswitch must ack Discord within 3s and the kill
    # switch already fails closed on RedisError, so a hung Redis must not block it.
    kill_switch = KillSwitch(
        Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
    )
    tracker = PortfolioTracker(session_factory, brokers, settings.trading_mode)
    finnhub = FinnhubClient(settings.finnhub_api_key) if settings.finnhub_api_key else None
    if finnhub is None:
        log.warning("provider_unconfigured", provider="finnhub", note="stock quotes skipped")
    fred = FredClient(settings.fred_api_key) if settings.fred_api_key else None
    if fred is None:
        log.warning("provider_unconfigured", provider="fred", note="VIX unknown - no size halving")

    if settings.anthropic_api_key:
        analyst = ClaudeAnalyst(settings, session_factory)
    else:
        analyst = stub_analyst
        log.warning("analyst_stubbed", note="no ANTHROPIC_API_KEY - holds only, no trades proposed")

    notifier = None
    if settings.discord_bot_token and settings.discord_channel_id:
        notifier = DiscordNotifier(settings.discord_bot_token, settings.discord_channel_id)
    else:
        log.warning(
            "notifier_unconfigured", note="no Discord token/channel - alerts go to logs only"
        )

    return CycleDeps(
        settings=settings,
        session_factory=session_factory,
        brokers=brokers,
        kill_switch=kill_switch,
        tracker=tracker,
        analyst=analyst,
        coingecko=coingecko,
        finnhub=finnhub,
        alpaca_data=alpaca_data,
        fred=fred,
        notifier=notifier,
        fear_greed=FearGreedClient(),
    )
