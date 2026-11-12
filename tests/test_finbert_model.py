"""The real FinBERT model. Runs only where the `sentiment` extra is installed:

    docker compose run --rm -v ./tests:/app/tests worker
        python -m pytest tests/test_finbert_model.py   (one command)

Skipped in the host venv, which never installs torch (D-026).
"""

import pytest

pytest.importorskip("transformers")

from trading_agent.ml.sentiment import FinBert


@pytest.fixture(scope="module")
def finbert() -> FinBert:
    """The pinned model, loaded once."""
    return FinBert()


def test_obvious_headlines_score_the_right_way(finbert):
    good, bad, flat = finbert(
        [
            "Company profits surge far above expectations",
            "Company files for bankruptcy after massive losses",
            "Company will hold its annual meeting on Tuesday",
        ]
    )
    assert good["positive"] > 0.5 > good["negative"]
    assert bad["negative"] > 0.5 > bad["positive"]
    assert flat["neutral"] > 0.5
    for scores in (good, bad, flat):
        assert set(scores) == {"positive", "negative", "neutral"}
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)


def test_batches_larger_than_one_step_keep_order(finbert):
    headlines = ["Shares soar on record revenue"] * 40 + ["Shares plunge on fraud probe"]
    scores = finbert(headlines)
    assert len(scores) == 41
    assert scores[-1]["negative"] > scores[-1]["positive"]
