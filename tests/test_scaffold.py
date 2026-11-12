"""Scaffold smoke tests: the package imports and the proposal boundary holds."""

import pydantic
import pytest


def test_package_imports():
    import trading_agent
    import trading_agent.brokers
    import trading_agent.data
    import trading_agent.graph.state
    import trading_agent.log
    import trading_agent.notify
    import trading_agent.risk
    import trading_agent.storage

    assert trading_agent.__version__


def test_trade_proposal_has_no_sizing_fields():
    """The LLM proposal schema must never grow sizing/risk fields (RISK_RULES.md)."""
    from trading_agent.graph.state import TradeProposal

    fields = set(TradeProposal.model_fields)
    assert fields == {"symbol", "asset_class", "action", "confidence", "reasoning"}


def test_trade_proposal_confidence_bounded():
    from trading_agent.graph.state import TradeProposal

    with pytest.raises(pydantic.ValidationError):
        TradeProposal(
            symbol="AAPL", asset_class="stock", action="buy", confidence=1.5, reasoning="x"
        )
