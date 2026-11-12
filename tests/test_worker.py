"""Background worker: schedule, broker isolation, and the retrain job's reporting."""

import pytest
import structlog

from trading_agent import worker
from trading_agent.config import Settings

META = {
    "version": "20260927T060000Z",
    "walk_forward": {
        "auc": 0.5296,
        "accuracy": 0.5363,
        "base_rate": 0.5571,
        "fold_auc": [0.53, 0.50, 0.56, 0.56, 0.50],
    },
}
SECRET = "sk-not-for-discord"


class _Notifier:
    sent: list[str] = []  # noqa: RUF012 - reset per test

    def __init__(self, token: str, channel: str) -> None:
        """Accept the real constructor's arguments."""

    def send(self, text: str) -> bool:
        """Record the message instead of posting it."""
        _Notifier.sent.append(text)
        return True


@pytest.fixture
def notifier(monkeypatch):
    """Discord configured, messages captured in _Notifier.sent."""
    _Notifier.sent = []
    monkeypatch.setattr(worker, "DiscordNotifier", _Notifier)
    monkeypatch.setattr(
        worker,
        "load_settings",
        lambda: Settings(discord_bot_token="t", discord_channel_id="1", _env_file=None),
    )
    return _Notifier


@pytest.fixture
def alpaca_keys(monkeypatch):
    """Placeholder Alpaca keys, so the news job builds its client without a local .env."""
    monkeypatch.setattr(
        worker,
        "load_settings",
        lambda: Settings(alpaca_api_key="k", alpaca_secret_key="s", _env_file=None),
    )


def test_retrain_is_scheduled_weekly_while_markets_are_closed():
    entry = worker.app.conf.beat_schedule["retrain-directional-model-weekly"]
    assert entry["task"] == worker.retrain_directional_model.name
    schedule = entry["schedule"]
    assert schedule.day_of_week == {0}  # Sunday
    assert schedule.hour == {6} and schedule.minute == {0}
    assert worker.app.conf.timezone == "UTC"


def test_broker_defaults_to_its_own_redis_db_apart_from_the_kill_switch():
    settings = Settings(_env_file=None)
    assert settings.celery_broker_url.endswith("/1")
    assert settings.redis_url.endswith("/0")


def test_retrain_reports_skill_in_plain_language(notifier, monkeypatch):
    monkeypatch.setattr(worker, "train_from_alpaca", lambda settings, root: META)
    assert worker.retrain_directional_model.apply().get() == META["version"]
    assert notifier.sent == [
        "[paper] prediction model retrained (20260927T060000Z): skill 0.530 (0.5 = coin flip), "
        "right 53.6% of the time vs 55.7% for always guessing up. "
        "logged from the next cycle (not shown to the analyst since D-032)."
    ]


def test_retrain_below_the_floor_says_it_is_not_used(notifier, monkeypatch):
    weak = {**META, "walk_forward": {**META["walk_forward"], "fold_auc": [0.5] * 5}}
    monkeypatch.setattr(worker, "train_from_alpaca", lambda settings, root: weak)
    worker.retrain_directional_model.apply()
    assert "NOT served" in notifier.sent[0]


def test_retrain_failure_pages_with_the_type_name_only_and_still_fails(notifier, monkeypatch):
    def boom(settings, root):
        """Fail the way a transport error would, URL and key in the text."""
        raise ConnectionError(f"https://data.alpaca.markets/?key={SECRET}")

    monkeypatch.setattr(worker, "train_from_alpaca", boom)
    with pytest.raises(ConnectionError):
        worker.retrain_directional_model.apply().get()
    assert notifier.sent == ["[paper] model retrain FAILED: ConnectionError"]


def test_retrain_without_discord_still_runs(monkeypatch):
    monkeypatch.setattr(worker, "load_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(worker, "train_from_alpaca", lambda settings, root: META)
    assert worker.retrain_directional_model.apply().get() == META["version"]


def test_a_hung_retrain_is_reported_before_the_hard_kill(notifier, monkeypatch):
    """The hard limit kills the process with no chance to report; the soft
    limit raises inside the task first, so Discord hears about the hang."""
    from celery.exceptions import SoftTimeLimitExceeded

    assert worker.app.conf.task_soft_time_limit < worker.app.conf.task_time_limit

    def hang(settings, root):
        """What Celery raises inside the task when the soft limit expires."""
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(worker, "train_from_alpaca", hang)
    with pytest.raises(SoftTimeLimitExceeded):
        worker.retrain_directional_model.apply().get()
    assert notifier.sent == ["[paper] model retrain FAILED: SoftTimeLimitExceeded"]


def test_retrain_without_alpaca_keys_reports_the_failure(notifier, monkeypatch):
    """No stub: the real train_from_alpaca refuses empty keys before any fetch."""
    monkeypatch.setattr(
        worker,
        "load_settings",
        lambda: Settings(
            discord_bot_token="t",
            discord_channel_id="1",
            alpaca_api_key="",
            alpaca_secret_key="",
            _env_file=None,
        ),
    )
    with pytest.raises(ValueError):
        worker.retrain_directional_model.apply().get()
    assert notifier.sent == ["[paper] model retrain FAILED: ValueError"]


def test_news_is_scored_every_ten_minutes():
    entry = worker.app.conf.beat_schedule["score-news"]
    assert entry["task"] == worker.score_news.name
    assert entry["schedule"].minute == set(range(0, 60, 10))


def test_a_failed_news_run_warns_and_waits_for_the_next(alpaca_keys, monkeypatch):
    """No Discord page every ten minutes during an outage: a warning, and the
    next run re-reads the same 24-hour window."""

    def boom(*args):
        """Fail like a news API outage."""
        raise ConnectionError(f"https://data.alpaca.markets/?key={SECRET}")

    monkeypatch.setattr(worker, "score_recent_news", boom)
    monkeypatch.setattr(worker, "_session_factory", lambda: None)
    monkeypatch.setattr(worker, "_finbert", lambda: None)
    with structlog.testing.capture_logs() as logs:
        assert worker.score_news.apply().get() == 0
    failed = [e for e in logs if e["event"] == "news_scoring_failed"]
    assert [(e["log_level"], e["reason"]) for e in failed] == [("warning", "ConnectionError")]


def test_a_successful_news_run_records_a_heartbeat_even_with_no_new_headlines(
    alpaca_keys, monkeypatch
):
    """The agent's staleness warning reads this heartbeat: a quiet news hour
    must not look like a dead worker."""
    marks: list[str] = []

    class _Session:
        """Minimal session context."""

        def __enter__(self):
            """Enter."""
            return self

        def __exit__(self, *exc):
            """Exit."""
            return False

        def commit(self):
            """Commit (no-op)."""

    class _Runs:
        """Records mark_success calls."""

        def __init__(self, session):
            """Bind nothing."""

        def mark_success(self, name, at):
            """Record the job name."""
            marks.append(name)

    monkeypatch.setattr(worker, "score_recent_news", lambda *args: 0)
    monkeypatch.setattr(worker, "_session_factory", lambda: _Session)
    monkeypatch.setattr(worker, "_finbert", lambda: None)
    monkeypatch.setattr(worker, "JobRunRepository", _Runs)
    assert worker.score_news.apply().get() == 0
    assert marks == ["score_news"]


def test_the_weekly_report_posts_monday_before_the_us_open(notifier, monkeypatch):
    entry = worker.app.conf.beat_schedule["weekly-report"]
    assert entry["task"] == worker.weekly_report.name
    assert entry["schedule"].day_of_week == {1}  # Monday
    assert entry["schedule"].hour == {13} and entry["schedule"].minute == {0}
    monkeypatch.setattr(worker, "_session_factory", lambda: None)
    monkeypatch.setattr(worker, "weekly_message", lambda settings, factory, now: "the report")
    assert worker.weekly_report.apply().get() == "the report"
    assert notifier.sent == ["the report"]


def test_a_failed_weekly_report_says_so_on_discord(notifier, monkeypatch):
    def boom(settings, factory, now):
        """Fail like a database outage."""
        raise ConnectionError(f"postgres://u:{SECRET}@h/db down")

    monkeypatch.setattr(worker, "_session_factory", lambda: None)
    monkeypatch.setattr(worker, "weekly_message", boom)
    assert worker.weekly_report.apply().get() == ""
    assert notifier.sent == ["[paper] weekly report FAILED: ConnectionError"]
