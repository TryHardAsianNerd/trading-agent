"""AgentState: the state threaded through the cycle, and the models it carries.

The risk gate turns `proposals` into `approved_orders`; LLM output never gets past
the proposal stage.
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class TradeProposal(BaseModel):
    """What the LLM analyst is allowed to produce — nothing more.

    No sizes, no stops, no risk parameters: those are computed by the
    deterministic risk engine, which the LLM cannot see or influence.
    """

    symbol: str
    asset_class: Literal["stock", "crypto"]
    action: Literal["buy", "sell", "hold"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ApprovedOrder(BaseModel):
    """Output of the risk gate: a proposal plus deterministic risk parameters.

    source_proposal is None only for the executor's stop-breach sales.
    """

    symbol: str
    asset_class: Literal["stock", "crypto"]
    side: Literal["buy", "sell"]
    quantity: Decimal
    stop_loss_price: Decimal | None
    source_proposal: TradeProposal | None


class AgentState(TypedDict, total=False):
    cycle_id: str
    cycle_started_at: datetime
    # False outside the NYSE session: stocks go unpriced, so no stock trade can pass the gate.
    stock_market_open: bool
    market_data: dict[str, object]
    features: dict[str, object]
    predictions: dict[str, object]
    proposals: list[TradeProposal]
    approved_orders: list[ApprovedOrder]
    rejections: list[dict[str, object]]
    stop_breaches: list[dict[str, object]]
    execution_results: list[dict[str, object]]
    # The scheduler's watchdog reads these two.
    portfolio_error: str
    reconcile_failed: list[str]
    notifications_sent: list[str]
