"""The swing sleeve's own portfolio, which the risk gate evaluates alone (D-040).

Reserve, position cap and breakers are measured on swing equity; risk per trade stays a
share of the whole account. So that deposits never read as returns, each snapshot stores a
time-weighted swing index, and an anchor is swing equity scaled by index_then / index_now.
"""

from dataclasses import replace
from decimal import Decimal

from trading_agent.portfolio.core import is_core, net_cost
from trading_agent.risk.limits import DEFAULT_LIMITS
from trading_agent.risk.models import PortfolioState
from trading_agent.storage.models import CashFlow, Trade

SWING_LIMITS = replace(DEFAULT_LIMITS, max_position_pct=Decimal("0.25"))
# The index never reaches 0: a wiped-out sleeve must read as a ~100% loss (anchors
# divide by it), and 0 x any later return would stay 0 - the breakers switched off.
INDEX_FLOOR = Decimal("1e-9")


def equity(
    portfolio: PortfolioState, marks: dict[str, Decimal], core_claim: Decimal
) -> tuple[Decimal, Decimal]:
    """Swing (cash, equity): stock venue cash less the core's claim, plus non-core holdings.

    Marks are the price or last close, never the cost, so a stock outside the session does
    not read as a loss. Cash may go negative; clamping it would make the next deposit a loss.
    """
    cash = portfolio.venue_cash.get("stock", Decimal(0)) - core_claim
    held = sum(
        (
            p.quantity * marks.get(s, p.avg_entry_price)
            for s, p in portfolio.positions.items()
            if not is_core(s)
        ),
        Decimal(0),
    )
    return cash, cash + held


def unexplained(stock_cash: Decimal, flows: list[CashFlow], stock_fills: list[Trade]) -> Decimal:
    """Stock venue cash that no ledger row or fill explains, such as a fee or a dividend."""
    deposited = sum(
        (f.amount for f in flows if f.kind == "deposit" and f.asset_class == "stock"), Decimal(0)
    )
    return stock_cash - (deposited - net_cost(stock_fills, flows[0].ts))


def next_index(
    previous: tuple[Decimal, Decimal] | None, swing_equity: Decimal, deposited: Decimal
) -> Decimal:
    """Advance the swing index from the previous (equity, index) by the return net of new deposits.

    Starts at 1, holds while the sleeve was empty, and never drops below INDEX_FLOOR.
    """
    if previous is None:
        return Decimal(1)
    before, index = previous
    if before <= 0:
        return index
    return max(INDEX_FLOOR, index * (swing_equity - deposited) / before)


def view(
    portfolio: PortfolioState,
    cash: Decimal,
    swing_equity: Decimal,
    index: Decimal,
    anchors: tuple[Decimal | None, Decimal | None, Decimal | None],
    loss_streak: int,
) -> PortfolioState:
    """The PortfolioState the gate evaluates: the swing sleeve alone.

    `anchors` is the swing index at day start, week start and peak; an unknown one reads as
    no loss. The whole account's equity becomes risk_equity, and crypto cash is all core.
    """

    def anchored(at: Decimal | None) -> Decimal:
        """Swing equity scaled by the index at the anchor over the index now."""
        if at is None:
            return swing_equity
        # A WIPED-OUT sleeve (index at the floor) still needs a positive anchor, or a
        # total loss reads as none; one that was simply never funded has no loss.
        base = INDEX_FLOOR if swing_equity <= 0 and index <= INDEX_FLOOR else swing_equity
        return base * at / index

    day, week, peak = anchors
    return PortfolioState(
        equity=swing_equity,
        cash=cash,
        peak_equity=anchored(peak),  # the peak index includes this cycle's
        day_start_equity=anchored(day),
        week_start_equity=anchored(week),
        consecutive_losses=loss_streak,
        positions={s: p for s, p in portfolio.positions.items() if not is_core(s)},
        untracked_positions={
            s: q for s, q in portfolio.untracked_positions.items() if not is_core(s)
        },
        venue_cash={"stock": cash, "crypto": Decimal(0)},
        risk_equity=portfolio.equity,
    )
