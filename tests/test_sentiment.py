"""News sentiment: the scoring job, the repository, and the feature join.

Needs the compose TimescaleDB (skipped without it, like the storage
integration tests). FinBERT itself is replaced by a fake scorer here: torch
is worker-only — test_finbert_model.py runs inside the worker image.
"""

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import structlog
from sqlalchemy import text

from trading_agent.config import Settings
from trading_agent.data.models import NewsArticle
from trading_agent.graph.deps import CycleDeps
from trading_agent.graph.nodes.feature_engine import compute_features
from trading_agent.ml.sentiment import MODEL_REVISION, score_recent_news
from trading_agent.storage.db import init_db, make_engine, make_session_factory
from trading_agent.storage.repos import JobRunRepository, NewsSentimentRepository

# Real time, not a fixed date: rows land in the live table (removed at teardown), and a
# future scored_at left behind by an interrupted run must not look like fresh data.
NOW = datetime.now(UTC)
# Symbols no real watchlist uses, so production rows never mix into these tests.
SYM, OTHER = "ZZSENT", "ZZOTHER"


@pytest.fixture(scope="module")
def session_factory():
    """Sessions on the compose database; test rows removed afterwards."""
    engine = make_engine(Settings(_env_file=None).database_url, connect_args={"connect_timeout": 3})
    try:
        init_db(engine)
    except Exception:
        pytest.skip("TimescaleDB not reachable - run: docker compose up -d")
    yield make_session_factory(engine)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM news_sentiment WHERE :a = ANY(symbols) OR :b = ANY(symbols)"),
            {"a": SYM, "b": OTHER},
        )


def _id() -> int:
    """An article id far above Alpaca's (~6e7) so tests never collide with real rows."""
    return random.randint(10**15, 10**16)


def _article(symbols: list[str], hours_ago: float = 1, headline: str = "h") -> NewsArticle:
    """A fetched article."""
    return NewsArticle(
        id=_id(),
        created_at=NOW - timedelta(hours=hours_ago),
        headline=headline,
        source="test",
        symbols=symbols,
    )


class _Client:
    """Serves canned articles as AlpacaDataClient.get_news would."""

    def __init__(self, articles: list[NewsArticle]) -> None:
        """Hold the articles to serve."""
        self.articles = articles
        self.calls: list[tuple] = []

    def get_news(self, symbols, start, end):
        """Record the request; return the articles."""
        self.calls.append((symbols, start, end))
        return self.articles


class _Scorer:
    """Positive for headlines containing 'up', negative otherwise; counts calls."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.seen: list[str] = []

    def __call__(self, headlines):
        """Score headlines deterministically."""
        self.seen.extend(headlines)
        return [
            {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
            if "up" in h
            else {"positive": 0.1, "negative": 0.7, "neutral": 0.2}
            for h in headlines
        ]


def test_new_headlines_are_scored_once_and_stored(session_factory):
    articles = [_article([SYM], headline="shares up"), _article([SYM, OTHER], headline="down")]
    client, scorer = _Client(articles), _Scorer()
    assert score_recent_news(client, session_factory, scorer, [SYM, OTHER], NOW) == 2
    assert client.calls == [([SYM, OTHER], NOW - timedelta(hours=24), NOW)]
    # The same window again: nothing new, the model is not even called.
    scorer2 = _Scorer()
    assert score_recent_news(client, session_factory, scorer2, [SYM, OTHER], NOW) == 0
    assert scorer2.seen == []
    with session_factory() as session:
        repo = NewsSentimentRepository(session)
        mean, count = repo.summary(SYM, NOW - timedelta(hours=24))
        assert count == 2
        assert mean == pytest.approx(((0.8 - 0.1) + (0.1 - 0.7)) / 2)
        assert repo.summary(OTHER, NOW - timedelta(hours=24)) == (pytest.approx(-0.6), 1)
        stored = session.execute(
            text("SELECT model_revision FROM news_sentiment WHERE article_id = :i"),
            {"i": articles[0].id},
        ).scalar()
        assert stored == MODEL_REVISION


def test_articles_without_watchlist_symbols_are_ignored(session_factory):
    scorer = _Scorer()
    assert score_recent_news(_Client([_article([])]), session_factory, scorer, [SYM], NOW) == 0
    assert scorer.seen == []


def test_summary_window_counts_only_recent_articles(session_factory):
    old = _article([SYM], hours_ago=30, headline="way up")
    score_recent_news(_Client([old]), session_factory, _Scorer(), [SYM], NOW)
    with session_factory() as session:
        repo = NewsSentimentRepository(session)
        _, recent = repo.summary(SYM, NOW - timedelta(hours=24))
        _, wider = repo.summary(SYM, NOW - timedelta(hours=48))
    assert wider == recent + 1
    assert NewsSentimentRepository(session_factory()).summary("ZZNONE", NOW) == (0.0, 0)


def _deps(session_factory) -> CycleDeps:
    """CycleDeps with only a session factory: enough for the feature engine."""
    return CycleDeps(
        settings=None,
        session_factory=session_factory,
        brokers=[],
        kill_switch=None,
        tracker=None,
        analyst=None,
        coingecko=None,
    )


def test_features_carry_sentiment_only_for_symbols_with_headlines(session_factory, monkeypatch):
    """Joined into the analyst's features; a symbol with no headlines gets no
    sentiment key at all rather than a misleading neutral 0."""
    score_recent_news(
        _Client([_article([SYM], headline="up up")]), session_factory, _Scorer(), [SYM], NOW
    )
    from trading_agent.graph.nodes import feature_engine

    monkeypatch.setattr(feature_engine, "snapshot", lambda candles: {"close": 1.0})
    state = {
        "cycle_id": "t",
        "cycle_started_at": NOW,
        "market_data": {"prices": {SYM: Decimal(1), "ZZNONE": Decimal(1)}},
    }
    features = compute_features(state, deps=_deps(session_factory))["features"]
    assert features[SYM]["news_count_24h"] >= 1.0
    assert -1.0 <= features[SYM]["news_sentiment_24h"] <= 1.0
    assert "news_sentiment_24h" not in features["ZZNONE"]
    assert all(type(v) is float for v in features[SYM].values())


@pytest.mark.parametrize(
    ("last_run", "warned"),
    [(None, True), (timedelta(minutes=61), True), (timedelta(minutes=20), False)],
)
def test_a_stopped_scoring_job_is_visible(session_factory, monkeypatch, last_run, warned):
    """Staleness is the worker's heartbeat, not the newest headline: a quiet
    news hour is fine, a job that stopped running is not."""
    from trading_agent.graph.nodes import feature_engine

    monkeypatch.setattr(feature_engine, "snapshot", lambda candles: {"close": 1.0})
    heartbeat = None if last_run is None else NOW - last_run
    monkeypatch.setattr(JobRunRepository, "last_success", lambda self, name: heartbeat)
    state = {"cycle_id": "t", "cycle_started_at": NOW, "market_data": {"prices": {SYM: 1}}}
    with structlog.testing.capture_logs() as logs:
        compute_features(state, deps=_deps(session_factory))
    stale = [e for e in logs if e["event"] == "news_sentiment_stale"]
    assert [e["log_level"] for e in stale] == (["warning"] if warned else [])


def test_a_sentiment_failure_keeps_the_price_features(session_factory, monkeypatch):
    """Sentiment is garnish: its failure must not cost the analyst the prices."""
    from trading_agent.graph.nodes import feature_engine

    def broken(self, symbol, since):
        """Fail like a missing table or an aborted transaction."""
        raise RuntimeError("relation news_sentiment does not exist")

    monkeypatch.setattr(feature_engine, "snapshot", lambda candles: {"close": 1.0})
    monkeypatch.setattr(NewsSentimentRepository, "summary", broken)
    state = {"cycle_id": "t", "cycle_started_at": NOW, "market_data": {"prices": {SYM: 1}}}
    with structlog.testing.capture_logs() as logs:
        features = compute_features(state, deps=_deps(session_factory))["features"]
    assert features == {SYM: {"close": 1.0}}
    unavailable = [e for e in logs if e["event"] == "news_sentiment_unavailable"]
    assert [e["log_level"] for e in unavailable] == ["warning"]


def test_heartbeat_round_trip(session_factory):
    with session_factory() as session:
        repo = JobRunRepository(session)
        assert repo.last_success("zz-test-job") is None
        repo.mark_success("zz-test-job", NOW - timedelta(hours=1))
        repo.mark_success("zz-test-job", NOW)
        assert repo.last_success("zz-test-job") == NOW
        session.rollback()  # never persisted
