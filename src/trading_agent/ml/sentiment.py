"""FinBERT sentiment scores on the watchlist's news headlines, run by the background worker.

torch and transformers (~1 GB) are imported inside FinBert, so the trading host never
loads them; it reads the stored scores. Only headlines are scored: FinBERT was tuned
on single financial sentences, and summaries mix several stories.

The model is pinned to one Hugging Face commit and loaded from its pytorch_model.bin
with torch's weights_only mode, which refuses code objects. Community safetensors
pull requests on the model repo are unvetted, so the worker image sets
DISABLE_SAFETENSORS_CONVERSION to keep transformers from fetching them.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from trading_agent.data.providers.alpaca import AlpacaDataClient
from trading_agent.log import get_logger
from trading_agent.storage.models import NewsSentiment
from trading_agent.storage.repos import NewsSentimentRepository

log = get_logger(__name__)

# Heartbeat name in job_runs (the agent warns when it goes quiet).
NEWS_JOB = "score_news"
MODEL_ID = "ProsusAI/finbert"
MODEL_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
# Re-read the last day every run; stored ids are skipped, so this only bounds the fetch.
FETCH_WINDOW = timedelta(hours=24)
BATCH_SIZE = 32

# headlines -> one {"positive", "negative", "neutral"} probability dict per headline.
Scorer = Callable[[list[str]], list[dict[str, float]]]


class FinBert:
    """The pinned FinBERT model, loaded once per worker process."""

    def __init__(self) -> None:
        """Download the model on first use (cached in the worker's volume) and load it.

        Raises ImportError where the `sentiment` extra is not installed (the host venv).
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        ).eval()
        self._labels = {int(i): name.lower() for i, name in self._model.config.id2label.items()}

    def __call__(self, headlines: list[str]) -> list[dict[str, float]]:
        """One {"positive", "negative", "neutral"} probability dict per headline, in order."""
        import torch

        scores: list[dict[str, float]] = []
        for i in range(0, len(headlines), BATCH_SIZE):
            batch = self._tokenizer(
                headlines[i : i + BATCH_SIZE],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                probs = torch.softmax(self._model(**batch).logits, dim=-1).tolist()
            scores.extend({self._labels[j]: p for j, p in enumerate(row)} for row in probs)
        return scores


def score_recent_news(
    client: AlpacaDataClient,
    session_factory: sessionmaker[Session],
    scorer: Scorer,
    symbols: list[str],
    now: datetime,
) -> int:
    """Score and store the last day's new watchlist headlines; returns how many were new."""
    articles = [a for a in client.get_news(symbols, now - FETCH_WINDOW, now) if a.symbols]
    with session_factory() as session:
        repo = NewsSentimentRepository(session)
        known = repo.known_ids([a.id for a in articles])
        new = [a for a in articles if a.id not in known]
        scores = scorer([a.headline for a in new]) if new else []
        repo.add(
            [
                NewsSentiment(
                    article_id=a.id,
                    created_at=a.created_at,
                    symbols=a.symbols,
                    headline=a.headline,
                    source=a.source,
                    positive=s["positive"],
                    negative=s["negative"],
                    neutral=s["neutral"],
                    model_revision=MODEL_REVISION,
                    scored_at=now,
                )
                for a, s in zip(new, scores, strict=True)
            ]
        )
        session.commit()
    log.info("news_scored", fetched=len(articles), scored=len(new))
    return len(new)
