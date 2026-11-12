"""Inputs and outputs of the risk gate.

PortfolioState and MarketEnvironment are assembled by deterministic code, never by the
LLM. The gate takes TradeProposals, the only thing the LLM produces, and returns
approved orders and rejections with machine-readable reasons.
"""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field

from trading_agent.graph.state import ApprovedOrder, TradeProposal


class Position(BaseModel):
    symbol: str
    asset_class: str  # authoritative venue routing — never taken from LLM output
    quantity: Decimal
    avg_entry_price: Decimal


class PortfolioState(BaseModel):
    equity: Decimal
    cash: Decimal
    peak_equity: Decimal
    day_start_equity: Decimal
    week_start_equity: Decimal
    consecutive_losses: int = Field(ge=0)
    positions: dict[str, Position] = {}
    # Held balances with no establishable entry price; exposure is unknown, not absent.
    untracked_positions: dict[str, Decimal] = {}
    # Cash per venue by asset class. Equity and the reserve are pooled, but a buy can only
    # spend its own venue's cash. Empty means pooled cash only.
    venue_cash: dict[str, Decimal] = {}
    # The equity that risk per trade is a share of; None means equity. The swing sleeve's
    # view sets the whole account's here.
    risk_equity: Decimal | None = None


class MarketEnvironment(BaseModel):
    """Layer-5 inputs, benign by default: a missing VIX close means no halving, not a halt."""

    vix: Decimal | None = None
    macro_event_pause: bool = False


class RejectReason(StrEnum):
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    DAILY_LOSS_BREAKER = "daily_loss_breaker"
    WEEKLY_LOSS_BREAKER = "weekly_loss_breaker"
    CONSECUTIVE_LOSSES_BREAKER = "consecutive_losses_breaker"
    BREAKER_COOLDOWN = "breaker_cooldown"
    MACRO_EVENT_PAUSE = "macro_event_pause"
    NO_POSITION_TO_SELL = "no_position_to_sell"
    # Buys open positions only, never add to one: the tested entry buys when flat (D-038).
    ALREADY_HELD = "already_held"
    INSUFFICIENT_CASH_AFTER_RESERVE = "insufficient_cash_after_reserve"
    POSITION_CAP_REACHED = "position_cap_reached"
    POSITION_SIZE_ZERO = "position_size_zero"
    DUPLICATE_PROPOSAL = "duplicate_proposal"
    INVALID_STOP = "invalid_stop"
    INSUFFICIENT_DATA = "insufficient_data"
    ILLIQUID = "illiquid"
    # The position breached its trail this cycle: the pending close outranks any proposal.
    STOP_BREACH_PENDING = "stop_breach_pending"
    EVALUATION_ERROR = "evaluation_error"


class Rejection(BaseModel):
    proposal: TradeProposal
    reason: RejectReason
    detail: str = ""


class GateDecision(BaseModel):
    approved: list[ApprovedOrder] = []
    rejected: list[Rejection] = []
    # Audit metadata for the decisions table; never consumed by trading logic.
    audit: dict[str, object] = {}
