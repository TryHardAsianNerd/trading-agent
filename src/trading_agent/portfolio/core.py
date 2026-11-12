"""The core sleeve: index funds and crypto bought with 70% of each deposit and held (D-040).

A holding sells in full when a complete daily close is CORE_TRAIL below its high since
opening, and is bought again only after a close at or below its 200-day line, then one
above. Core and swing never share a symbol, so a position's sleeve is its symbol.
"""

from datetime import datetime
from decimal import Decimal

from trading_agent.data.models import Candle
from trading_agent.graph.state import ApprovedOrder
from trading_agent.log import get_logger
from trading_agent.portfolio.ledger import CRYPTO_SHARE
from trading_agent.portfolio.tracker import average_cost_walk, day_start
from trading_agent.risk.limits import DEFAULT_LIMITS
from trading_agent.risk.models import PortfolioState
from trading_agent.storage.models import CashFlow, Order, Trade

log = get_logger(__name__)

# Target shares of the whole account.
CORE_TARGETS = {
    "VOO": Decimal("0.375"),
    "QQQM": Decimal("0.225"),
    "BTC-USD": Decimal("0.06"),
    "ETH-USD": Decimal("0.04"),
}
CORE_STOCKS = ["VOO", "QQQM"]
CORE_CRYPTO = ["BTC-USD", "ETH-USD"]
CORE_SHARE = sum(CORE_TARGETS.values(), Decimal(0))  # 70% of each deposit
CRYPTO_CAP = CRYPTO_SHARE  # the 10% cap is also the crypto venue's share of each deposit
CORE_TRAIL = Decimal("0.25")
TREND_DAYS = 200
# Above Alpaca's $1 minimum so the 1% headroom left by venue_cash_usable_pct
# does not become a second round of dust orders.
MIN_ORDER = DEFAULT_LIMITS.min_order_value
CANDLES_NEEDED = 400  # the 200-day line on every day since a sale, with room to spare


def is_core(symbol: str) -> bool:
    """Whether a symbol belongs to the core sleeve."""
    return symbol in CORE_TARGETS


def allocate(
    cash: Decimal,
    weights: dict[str, Decimal],
    values: dict[str, Decimal],
    eligible: set[str],
) -> dict[str, Decimal]:
    """Split cash across holdings in proportion to their shortfalls below target weight.

    `values` holds each symbol's value plus pending buys. An ineligible holding keeps its
    share in cash, and amounts under MIN_ORDER are dropped.
    """
    total_weight = sum(weights.values(), Decimal(0))
    total = cash + sum((values.get(s, Decimal(0)) for s in weights), Decimal(0))
    shortfall = {
        s: max(Decimal(0), w / total_weight * total - values.get(s, Decimal(0)))
        for s, w in weights.items()
    }
    need = sum(shortfall.values(), Decimal(0))
    if need <= 0:
        return {}
    alloc = {s: shortfall[s] * cash / need for s in weights}
    return {s: a for s, a in alloc.items() if s in eligible and a >= MIN_ORDER}


def average_closes(bars: list[Candle], days: int = TREND_DAYS) -> list[Decimal | None]:
    """The trailing `days`-bar average close at each bar, None until enough bars exist."""
    out: list[Decimal | None] = []
    running = Decimal(0)
    for i, bar in enumerate(bars):
        running += bar.close
        if i >= days:
            running -= bars[i - days].close
        out.append(running / days if i >= days - 1 else None)
    return out


def crossed_back(bars: list[Candle], sold_at: datetime) -> bool:
    """Whether a close at or below the 200-day line since the sale was followed by one above."""
    seen_below = False
    for bar, line in zip(bars, average_closes(bars), strict=True):
        if line is None or bar.ts < day_start(sold_at):
            continue
        if bar.close <= line:
            seen_below = True
        elif seen_below:
            return True
    return False


def trail_breached(
    close: Decimal, avg_cost: Decimal, closes_since_open: list[Decimal], trail: Decimal = CORE_TRAIL
) -> bool:
    """Whether a close is `trail` or more below the high since opening, floored at avg_cost."""
    return close <= max([avg_cost, *closes_since_open]) * (1 - trail)


def budgets(
    stock_claim: Decimal, venue_cash: dict[str, Decimal], crypto_value: Decimal, equity: Decimal
) -> dict[str, Decimal]:
    """What the core may spend at each venue now, after the venue_cash_usable_pct headroom.

    Cash never moves between venues, and crypto buys stop at CRYPTO_CAP of the account.
    """
    usable = DEFAULT_LIMITS.venue_cash_usable_pct
    crypto = min(
        max(Decimal(0), venue_cash.get("crypto", Decimal(0))) * usable,
        max(Decimal(0), CRYPTO_CAP * equity - crypto_value),
    )
    stock = max(Decimal(0), min(stock_claim, venue_cash.get("stock", Decimal(0)))) * usable
    return {"stock": stock, "crypto": crypto}


def venue_buys(
    cash: dict[str, Decimal], values: dict[str, Decimal], eligible: set[str]
) -> dict[str, Decimal]:
    """Each venue's budget spread over its core symbols toward their targets."""
    out: dict[str, Decimal] = {}
    for venue, group in (("crypto", CORE_CRYPTO), ("stock", CORE_STOCKS)):
        weights = {s: CORE_TARGETS[s] for s in group}
        out |= allocate(cash[venue], weights, values, eligible)
    return out


def reserved(open_orders: list[Order], prices: dict[str, Decimal]) -> dict[str, Decimal]:
    """What each symbol's open core buys will cost at today's price (0 without a price)."""
    out: dict[str, Decimal] = {}
    for o in open_orders:
        if o.side == "buy":
            cost = o.quantity * prices.get(o.symbol, Decimal(0))
            out[o.symbol] = out.get(o.symbol, Decimal(0)) + cost
    return out


def net_cost(fills: list[Trade], start: datetime) -> Decimal:
    """What fills since `start` took out of cash: buys plus fees, less sells after fees."""
    spent = Decimal(0)
    for f in fills:
        if f.filled_at < start:
            continue
        notional = f.quantity * f.fill_price
        spent += notional + f.fees if f.side == "buy" else -(notional - f.fees)
    return spent


def stock_core_cash(flows: list[CashFlow], fills: list[Trade], reserved: Decimal) -> Decimal:
    """The stock venue cash that belongs to the core: its deposits, net of fills and open buys."""
    deposited = sum((f.core_amount for f in flows if f.asset_class == "stock"), Decimal(0))
    return deposited - net_cost(fills, flows[0].ts) - reserved


def plan(
    portfolio: PortfolioState,
    prices: dict[str, Decimal],
    candles: dict[str, list[Candle]],
    fills: dict[str, list[Trade]],
    flows: list[CashFlow],
    open_orders: list[Order],
    now: datetime,
) -> tuple[list[ApprovedOrder], dict[str, object]]:
    """This cycle's core orders (stop sells, then deposit buys) and an audit dict.

    A symbol without a price this cycle is neither sold nor bought but still counts toward
    the weights. An open sell counts as a sale already made, since its fill is booked later.
    """
    orders: list[ApprovedOrder] = []
    audit: dict[str, object] = {"sells": [], "buys": {}, "waiting": []}
    values: dict[str, Decimal] = {}
    eligible: set[str] = set()
    pending = reserved(open_orders, prices)
    selling = {o.symbol for o in open_orders if o.side == "sell"}
    for symbol in CORE_TARGETS:
        price = prices.get(symbol)
        bars = [c for c in candles.get(symbol, []) if c.ts < day_start(now)]
        held = portfolio.positions.get(symbol)
        history = [f for f in fills.get(symbol, []) if f.filled_at >= flows[0].ts]
        values[symbol] = pending.get(symbol, Decimal(0))
        if symbol in selling:
            audit["waiting"].append(symbol)  # sold; the 200-day cross decides from here
            continue
        if held is not None:
            values[symbol] += held.quantity * (price or held.avg_entry_price)
            _, _, opened = average_cost_walk(history)
            if price is not None and not bars:
                log.warning("core_trail_unchecked", symbol=symbol, reason="no complete bars")
            if price is not None and bars:
                # No fill history (a position the book missed): trail from the cost alone.
                since = [c.close for c in bars if opened and c.ts >= day_start(opened)]
                if trail_breached(bars[-1].close, held.avg_entry_price, since):
                    orders.append(_order(symbol, "sell", held.quantity, held.asset_class))
                    high = max([held.avg_entry_price, *since])
                    audit["sells"].append(
                        {"symbol": symbol, "close": str(bars[-1].close), "high": str(high)}
                    )
                    continue
        else:
            sells = [f.filled_at for f in history if f.side == "sell"]
            if sells and not crossed_back(bars, max(sells)):
                audit["waiting"].append(symbol)
                continue
        if price is not None:
            eligible.add(symbol)

    claim = stock_core_cash(
        flows,
        [f for s in CORE_STOCKS for f in fills.get(s, [])],
        sum((pending.get(s, Decimal(0)) for s in CORE_STOCKS), Decimal(0)),
    )
    crypto_value = sum((values[s] for s in CORE_CRYPTO), Decimal(0))
    cash = budgets(claim, portfolio.venue_cash, crypto_value, portfolio.equity)
    audit["cash"] = {venue: str(amount) for venue, amount in cash.items()}
    for symbol, dollars in venue_buys(cash, values, eligible).items():
        asset_class = "crypto" if symbol in CORE_CRYPTO else "stock"
        orders.append(_order(symbol, "buy", dollars / prices[symbol], asset_class))
        audit["buys"][symbol] = str(dollars)
    return orders, audit


def _order(symbol: str, side: str, quantity: Decimal, asset_class: str) -> ApprovedOrder:
    """A core order: no venue stop, no analyst proposal behind it."""
    return ApprovedOrder(
        symbol=symbol,
        asset_class=asset_class,
        side=side,
        quantity=quantity,
        stop_loss_price=None,
        source_proposal=None,
    )
