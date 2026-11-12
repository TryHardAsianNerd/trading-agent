"""The risk gate: LLM proposals in, deterministic verdicts out.

Checks run most severe first:
  1. Layer 4 drawdown, first so the kill switch arms even during a cooldown
  2. Kill switch / cooldown (Redis, fail-closed)
  3. Layer 3 breakers: reject and arm a cooldown (weekly: the rest of the week)
  4. Layer 5: macro pause rejects; VIX above the threshold halves buy size
  5. Each proposal, with batch accounting. Buys hit the liquidity filter before any
     sizing, so a rejected symbol uses no cash or headroom. Sells reduce risk and are
     never liquidity-filtered.

The LLM decides how many proposals it emits, so limits hold across the whole batch:
one proposal per symbol, no buying a symbol already held, a 10% cap per new position,
and approved buys draw down running remainders of both deployable cash (so the batch
never dips below the 20% reserve) and their own venue's cash (the reserve is pooled,
but only the executing venue can pay).

Only symbol and intent are taken from the LLM: asset_class comes from the position or
market data, and confidence/reasoning are kept for audit. A failure evaluating one
proposal rejects that proposal only; it never aborts the batch or escapes the gate.
"""

from datetime import datetime
from decimal import Decimal

from trading_agent.data.models import Candle
from trading_agent.graph.state import ApprovedOrder, TradeProposal
from trading_agent.log import get_logger
from trading_agent.risk.breakers import seconds_until_week_end, tripped_breaker
from trading_agent.risk.killswitch import AUTO_KILL_SWITCH_REASON, KillSwitch
from trading_agent.risk.limits import DEFAULT_LIMITS, RiskLimits
from trading_agent.risk.liquidity import is_liquid
from trading_agent.risk.models import (
    GateDecision,
    MarketEnvironment,
    PortfolioState,
    Rejection,
    RejectReason,
)
from trading_agent.risk.sizing import deployable_cash, sized_quantity
from trading_agent.risk.stops import atr_stop_price, wilder_atr

log = get_logger(__name__)

_COOLDOWN_BREAKERS = {
    RejectReason.DAILY_LOSS_BREAKER,
    RejectReason.WEEKLY_LOSS_BREAKER,
    RejectReason.CONSECUTIVE_LOSSES_BREAKER,
}


def evaluate(
    proposals: list[TradeProposal],
    portfolio: PortfolioState,
    candles_by_symbol: dict[str, list[Candle]],
    prices: dict[str, Decimal],
    environment: MarketEnvironment,
    kill_switch: KillSwitch,
    limits: RiskLimits = DEFAULT_LIMITS,
    now: datetime | None = None,
) -> GateDecision:
    """Run the risk gate over one cycle's proposals; never raises.

    Holds are audited, not evaluated. `now` times the weekly breaker's pause: None is the
    current time, and the backtest passes its simulated clock.
    """
    decision = GateDecision()
    actionable = [p for p in proposals if p.action != "hold"]
    decision.audit["holds"] = [p.symbol for p in proposals if p.action == "hold"]

    def reject_all(reason: RejectReason, detail: str = "") -> GateDecision:
        decision.rejected = [
            Rejection(proposal=p, reason=reason, detail=detail) for p in actionable
        ]
        return decision

    # Layer 4 first: the kill switch must arm even mid-cooldown; others read it independently.
    breaker = tripped_breaker(portfolio, limits)
    if breaker is RejectReason.MAX_DRAWDOWN_EXCEEDED:
        # Arm it once: cycles keep running under it (the core and the stops), and re-arming
        # would page every cycle or overwrite a human's manual switch.
        if not kill_switch.is_halted()[1].startswith("kill_switch:"):
            kill_switch.activate(AUTO_KILL_SWITCH_REASON)
        return reject_all(breaker, "kill switch activated")

    halted, halt_reason = kill_switch.is_halted()
    if halted:
        reason = (
            RejectReason.BREAKER_COOLDOWN
            if halt_reason.startswith("cooldown:")
            else RejectReason.KILL_SWITCH_ACTIVE
        )
        return reject_all(reason, halt_reason)

    if breaker in _COOLDOWN_BREAKERS:
        ttl = (
            seconds_until_week_end(now)
            if breaker is RejectReason.WEEKLY_LOSS_BREAKER
            else None  # daily/consecutive: default 1h minimum
        )
        kill_switch.start_cooldown(breaker.value, limits, ttl_seconds=ttl)
        return reject_all(breaker)

    if environment.macro_event_pause:
        return reject_all(RejectReason.MACRO_EVENT_PAUSE)

    vix_halve = environment.vix is not None and environment.vix > limits.vix_reduce_threshold
    if vix_halve:
        decision.audit["vix_halved"] = True
    if portfolio.untracked_positions:
        # Unquantifiable exposure: audit it, and never add to a position whose size we can't see.
        decision.audit["untracked_positions"] = {
            s: str(q) for s, q in portfolio.untracked_positions.items()
        }

    remaining_cash = deployable_cash(portfolio, limits)
    remaining_venue = {
        ac: cash * limits.venue_cash_usable_pct for ac, cash in portfolio.venue_cash.items()
    }
    batch_symbol_value: dict[str, Decimal] = {}
    seen_symbols: set[str] = set()

    for proposal in actionable:
        symbol = proposal.symbol
        if symbol in seen_symbols:
            decision.rejected.append(
                Rejection(
                    proposal=proposal,
                    reason=RejectReason.DUPLICATE_PROPOSAL,
                    detail="one proposal per symbol per cycle",
                )
            )
            continue
        seen_symbols.add(symbol)

        try:
            outcome = _evaluate_one(
                proposal,
                portfolio,
                candles_by_symbol,
                prices,
                vix_halve,
                remaining_cash,
                remaining_venue,
                batch_symbol_value.get(symbol, Decimal(0)),
                limits,
            )
        except Exception as e:  # fail closed per proposal, never abort the batch
            log.exception("risk_gate_proposal_evaluation_failed", symbol=symbol)
            outcome = Rejection(
                proposal=proposal,
                reason=RejectReason.EVALUATION_ERROR,
                detail=f"{type(e).__name__}",
            )

        if isinstance(outcome, Rejection):
            decision.rejected.append(outcome)
        else:
            decision.approved.append(outcome)
            if outcome.side == "buy":
                notional = outcome.quantity * prices[symbol]
                remaining_cash -= notional
                if outcome.asset_class in remaining_venue:
                    remaining_venue[outcome.asset_class] -= notional
                batch_symbol_value[symbol] = batch_symbol_value.get(symbol, Decimal(0)) + notional

    illiquid = {
        r.proposal.symbol: r.detail for r in decision.rejected if r.reason is RejectReason.ILLIQUID
    }
    if illiquid:
        decision.audit["illiquid"] = illiquid

    log.info(
        "risk_gate_evaluated",
        approved=[o.symbol for o in decision.approved],
        rejected=[(r.proposal.symbol, r.reason.value) for r in decision.rejected],
        audit=decision.audit,
    )
    return decision


def _position_value(portfolio: PortfolioState, symbol: str, prices: dict[str, Decimal]) -> Decimal:
    """Market value of the held position, at its entry price when unpriced; 0 when none."""
    position = portfolio.positions.get(symbol)
    if position is None:
        return Decimal(0)
    price = prices.get(symbol, position.avg_entry_price)
    return position.quantity * price


def _evaluate_one(
    proposal: TradeProposal,
    portfolio: PortfolioState,
    candles_by_symbol: dict[str, list[Candle]],
    prices: dict[str, Decimal],
    vix_halve: bool,
    remaining_cash: Decimal,
    remaining_venue: dict[str, Decimal],
    batch_value_this_symbol: Decimal,
    limits: RiskLimits,
) -> ApprovedOrder | Rejection:
    """Approve or reject one buy or sell; unexpected errors propagate to the caller.

    A sell closes the whole position with no stop. A buy must clear the liquidity filter,
    then gets an ATR stop and a Layer-2 size. An asset class missing from remaining_venue
    is limited by remaining_cash only.
    """
    symbol = proposal.symbol

    if proposal.action == "sell":
        position = portfolio.positions.get(symbol)
        if position is None or position.quantity <= 0:
            return Rejection(proposal=proposal, reason=RejectReason.NO_POSITION_TO_SELL)
        # Risk-reducing full close, no stop; asset_class from the position, never the proposal.
        return ApprovedOrder(
            symbol=symbol,
            asset_class=position.asset_class,
            side="sell",
            quantity=position.quantity,
            stop_loss_price=None,
            source_proposal=proposal,
        )

    held = portfolio.positions.get(symbol)
    if held is not None and held.quantity > 0:
        return Rejection(proposal=proposal, reason=RejectReason.ALREADY_HELD)
    if symbol in portfolio.untracked_positions:
        return Rejection(
            proposal=proposal,
            reason=RejectReason.INSUFFICIENT_DATA,
            detail="held exposure unquantifiable this cycle - refusing to add to it",
        )
    price = prices.get(symbol)
    candles = candles_by_symbol.get(symbol, [])
    if price is None or price <= 0:
        return Rejection(
            proposal=proposal, reason=RejectReason.INSUFFICIENT_DATA, detail="no current price"
        )
    if not candles:
        return Rejection(
            proposal=proposal, reason=RejectReason.INSUFFICIENT_DATA, detail="no candles"
        )
    asset_class = candles[0].asset_class  # data-derived, not LLM-derived

    # Liquidity is a property of the instrument, so it is settled before stops or sizing.
    try:
        liquid, liquidity_detail = is_liquid(candles, asset_class, limits)
    except ValueError as e:
        return Rejection(proposal=proposal, reason=RejectReason.INSUFFICIENT_DATA, detail=str(e))
    if not liquid:
        return Rejection(proposal=proposal, reason=RejectReason.ILLIQUID, detail=liquidity_detail)

    try:
        atr = wilder_atr(candles)
    except ValueError as e:
        return Rejection(proposal=proposal, reason=RejectReason.INSUFFICIENT_DATA, detail=str(e))
    if atr <= 0:
        return Rejection(
            proposal=proposal,
            reason=RejectReason.INSUFFICIENT_DATA,
            detail="zero volatility (flat candles) — cannot place a meaningful stop",
        )

    stop = atr_stop_price(price, atr, limits)
    if stop <= 0:
        return Rejection(
            proposal=proposal,
            reason=RejectReason.INVALID_STOP,
            detail=f"ATR stop {stop} not positive (volatility too high for entry {price})",
        )

    existing_value = _position_value(portfolio, symbol, prices) + batch_value_this_symbol
    available = min(remaining_cash, remaining_venue.get(asset_class, remaining_cash))
    try:
        quantity = sized_quantity(
            portfolio,
            price,
            stop,
            limits,
            existing_position_value=existing_value,
            available_cash=available,
        )
    except ValueError as e:
        return Rejection(proposal=proposal, reason=RejectReason.INVALID_STOP, detail=str(e))
    if vix_halve:
        quantity /= 2
    if asset_class == "stock" and 0 < quantity * price < limits.min_order_value:
        return Rejection(
            proposal=proposal,
            reason=RejectReason.POSITION_SIZE_ZERO,
            detail=f"${quantity * price:.2f} is under the ${limits.min_order_value} minimum order",
        )
    if quantity <= 0:
        if existing_value >= portfolio.equity * limits.max_position_pct:
            reason = RejectReason.POSITION_CAP_REACHED
        elif available <= 0:
            reason = RejectReason.INSUFFICIENT_CASH_AFTER_RESERVE
        else:
            reason = RejectReason.POSITION_SIZE_ZERO
        return Rejection(proposal=proposal, reason=reason)

    return ApprovedOrder(
        symbol=symbol,
        asset_class=asset_class,
        side="buy",
        quantity=quantity,
        stop_loss_price=stop,
        source_proposal=proposal,
    )
