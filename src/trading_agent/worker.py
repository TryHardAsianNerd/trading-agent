"""Celery worker: slow jobs kept out of the trading process, in their own container.

    celery -A trading_agent.worker worker --beat --concurrency=1

Jobs: the model retrain (Sundays), the weekly report (Mondays) and FinBERT news scoring
(every 10 minutes). Results are not stored; each job reports through the logs and Discord.
"""

import functools
import logging
from datetime import UTC, datetime

from celery import Celery, signals
from celery.schedules import crontab
from sqlalchemy.orm import Session, sessionmaker

from trading_agent.config import load_settings
from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.log import configure_logging, get_logger
from trading_agent.ml import model
from trading_agent.ml.sentiment import NEWS_JOB, FinBert, score_recent_news
from trading_agent.ml.train import train_from_alpaca
from trading_agent.notify.discord import DiscordNotifier
from trading_agent.reports.weekly import weekly_message
from trading_agent.storage.db import init_db, make_engine, make_session_factory
from trading_agent.storage.repos import JobRunRepository

log = get_logger(__name__)

app = Celery("trading_agent", broker=load_settings().celery_broker_url)
app.conf.update(
    timezone="UTC",
    task_ignore_result=True,
    # A retrain takes seconds, so this means hung (alpaca-py sets no request timeout). The
    # soft limit raises inside the task so Discord hears of it; the hard limit is a backstop.
    task_soft_time_limit=25 * 60,
    task_time_limit=30 * 60,
    beat_schedule={
        "retrain-directional-model-weekly": {
            "task": "trading_agent.retrain_directional_model",
            "schedule": crontab(minute=0, hour=6, day_of_week="sunday"),
        },
        "weekly-report": {
            "task": "trading_agent.weekly_report",
            # Monday 13:00 UTC: the week (and its Sunday crypto close) is complete,
            # 30 minutes before the US open.
            "schedule": crontab(minute=0, hour=13, day_of_week="monday"),
        },
        "score-news": {
            "task": "trading_agent.score_news",
            "schedule": crontab(minute="*/10"),
        },
    },
)


@signals.setup_logging.connect
def _setup_logging(loglevel: int = logging.INFO, **_: object) -> None:
    """Route Celery's logging through the app's JSON stream and secret redaction."""
    configure_logging(loglevel)


@app.task(name="trading_agent.retrain_directional_model")
def retrain_directional_model() -> str:
    """Retrain the directional model, report it on Discord and return the new version id."""
    settings = load_settings()
    notifier = None
    if settings.discord_bot_token and settings.discord_channel_id:
        notifier = DiscordNotifier(settings.discord_bot_token, settings.discord_channel_id)
    try:
        meta = train_from_alpaca(settings, model.MODEL_DIR)
    except Exception as e:
        log.exception("model_retrain_failed")
        if notifier is not None:
            notifier.send(f"[{settings.trading_mode}] model retrain FAILED: {type(e).__name__}")
        raise
    if notifier is not None:
        notifier.send(format_retrain_notice(meta, settings.trading_mode))
    return meta["version"]


def format_retrain_notice(meta: dict, trading_mode: str) -> str:
    """Plain-language summary of a finished retrain."""
    wf = meta["walk_forward"]
    used = (
        "logged from the next cycle (not shown to the analyst since D-032)"
        if model.is_servable(wf)
        else "NOT served - below the skill floor, no predictions are logged"
    )
    return (
        f"[{trading_mode}] prediction model retrained ({meta['version']}): "
        f"skill {wf['auc']:.3f} (0.5 = coin flip), "
        f"right {wf['accuracy']:.1%} of the time vs {wf['base_rate']:.1%} for always guessing up. "
        f"{used}."
    )


@functools.cache
def _session_factory() -> sessionmaker[Session]:
    """The worker's database sessions, created (and the schema ensured) once."""
    engine = make_engine(load_settings().database_url)
    init_db(engine)  # the worker may start before the agent has ever created the tables
    return make_session_factory(engine)


@functools.cache
def _finbert() -> FinBert:
    """FinBERT, loaded once per worker process (the load takes seconds)."""
    return FinBert()


@app.task(name="trading_agent.score_news")
def score_news() -> int:
    """Score the watchlist's new headlines with FinBERT; returns how many, 0 on failure.

    A failure is a warning, not a page: the next run retries the same window.
    """
    settings = load_settings()
    try:
        client = AlpacaDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
        sessions = _session_factory()
        scored = score_recent_news(
            client,
            sessions,
            _finbert(),
            settings.stock_symbols + settings.crypto_symbols,
            datetime.now(UTC),
        )
        # Heartbeat even when nothing new arrived: "no news" is not "job dead".
        with sessions() as session:
            JobRunRepository(session).mark_success(NEWS_JOB, datetime.now(UTC))
            session.commit()
        return scored
    except Exception as e:
        log.warning("news_scoring_failed", reason=type(e).__name__, exc_info=True)
        return 0


@app.task(name="trading_agent.weekly_report")
def weekly_report() -> str:
    """Post last week's report to Discord and return it; "" when building it failed."""
    settings = load_settings()
    notifier = None
    if settings.discord_bot_token and settings.discord_channel_id:
        notifier = DiscordNotifier(settings.discord_bot_token, settings.discord_channel_id)
    try:
        message = weekly_message(settings, _session_factory(), datetime.now(UTC))
    except Exception as e:
        log.warning("weekly_report_failed", reason=type(e).__name__, exc_info=True)
        if notifier is not None:
            notifier.send(f"[{settings.trading_mode}] weekly report FAILED: {type(e).__name__}")
        return ""
    log.info("weekly_report", message=message)
    if notifier is not None:
        notifier.send(message)
    return message
