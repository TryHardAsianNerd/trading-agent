"""The risk step's account drawdown alert (D-047): pages once, never pauses."""

from contextlib import nullcontext
from decimal import Decimal as D
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from trading_agent.graph.nodes import risk_manager
from trading_agent.risk.models import PortfolioState


def _account(equity: str, peak: str = "1000") -> PortfolioState:
    """The whole account at `equity` below a deposit-adjusted `peak`."""
    return PortfolioState(
        equity=D(equity),
        cash=D(0),
        peak_equity=D(peak),
        day_start_equity=D(equity),
        week_start_equity=D(equity),
        consecutive_losses=0,
    )


def _deps(monkeypatch, previous: tuple[str, str] | None) -> SimpleNamespace:
    """Cycle deps whose newest snapshot is `previous` (equity, peak_equity), or none."""
    snap = None
    if previous is not None:
        snap = SimpleNamespace(equity=D(previous[0]), peak_equity=D(previous[1]))
    repo = SimpleNamespace(latest=lambda mode, since: snap)
    monkeypatch.setattr(risk_manager, "SnapshotRepository", lambda session: repo)
    return SimpleNamespace(
        session_factory=lambda: nullcontext(None),
        tracker=SimpleNamespace(trading_mode="paper"),
    )


def _alerts(deps, account: PortfolioState) -> list[dict]:
    """Run the check and return the account_drawdown_alert events it logged."""
    with capture_logs() as logs:
        risk_manager._alert_account_drawdown(deps, account, None)
    return [e for e in logs if e["event"] == "account_drawdown_alert"]


@pytest.mark.parametrize(
    ("previous", "equity", "pages"),
    [
        (("900", "1000"), "840", True),  # 10% -> 16%: the crossing pages
        (None, "850", True),  # the first cycle already 15% down pages (the limit counts)
        (("840", "1000"), "830", False),  # still below: paged last cycle, no repeat
        (("900", "1000"), "860", False),  # 14%: above the line
    ],
)
def test_the_alert_pages_once_on_the_crossing(monkeypatch, previous, equity, pages):
    alerts = _alerts(_deps(monkeypatch, previous), _account(equity))
    assert bool(alerts) is pages
    if pages:
        (alert,) = alerts
        assert alert["log_level"] == "error"  # the notifier posts error-level events
        assert "alert only, nothing paused" in alert["reason"]


def test_a_failing_check_never_fails_the_risk_step(monkeypatch):
    def broken(session):
        """Fail like a database outage."""
        raise RuntimeError("db down")

    deps = _deps(monkeypatch, None)
    monkeypatch.setattr(risk_manager, "SnapshotRepository", broken)
    with capture_logs() as logs:
        risk_manager._alert_account_drawdown(deps, _account("800"), None)
    assert [e["event"] for e in logs] == ["account_drawdown_alert_failed"]
