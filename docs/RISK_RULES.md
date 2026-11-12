# Risk Rules — Frozen

These are the non-negotiable trading safety rules from the blueprint. **Human-only changes.** No agent may weaken, bypass, or make these configurable-by-LLM. They are implemented as pure deterministic functions in `src/trading_agent/risk/`.

## Layer 1 — Per-trade stop-losses

- Every swing position gets an ATR-based stop, `stop_price = entry − (N × ATR)`, N between 1.5 and 3.0, held by the app: each cycle a position whose price is at or below its stop is sold in full. The sale goes out under every halt except the manual kill switch and an unreachable Redis (Jason, 2026-10-10, D-054). There is no swing stop while the agent is not running.
- Every core holding (VOO, QQQM, BTC, ETH) sells in full when a daily close is 25% or more below its highest close since it was bought (floored at cost), checked by the app on complete closes; it is bought again only after its price crosses back above its 200-day average (Jason, 2026-09-24, D-040)
- ~~Trailing stops (3% trail) to lock in profits on winners~~ — removed by Jason, 2026-09-23 (D-032): on daily bars the fixed 3% trail sold ordinary noise and was the largest single loss driver in every backtested strategy (D-031). The ATR stop (N = 3.0) is the exit rule; any future trail must be justified by the backtest first.

## Layer 2 — Position sizing

- Risk per trade: 1–2% of portfolio, fixed-fractional:
  `position_size = (portfolio × risk_pct) / (entry − stop_loss)`
- No single position may exceed 5–10% of total portfolio value (live: 10%, Jason, 2026-09-23, D-032)
- With the core + swing plan (D-040; swing restored by D-048): 70% of every deposit buys the core toward its target weights (VOO 37.5%, QQQM 22.5%, BTC 6%, ETH 4% of the account), which is exempt from the per-position cap; a swing position risks 1% of the account, is capped at 25% of swing-sleeve equity, and swing buys are fractional, at least $10 (Jason, 2026-10-10, D-054)
- Crypto never exceeds 10% of the account through deposits (nothing is sold to trim it)

## Layer 3 — Circuit breakers

With the core plan (D-040, D-042) these measure the swing sleeve's own returns and stop only swing entries; core buying continues and every stop keeps working.

- Halt all trading if daily losses exceed 3–5% of portfolio
- Pause for the remainder of the week at 5–7% weekly drawdown
- Minimum 1-hour cooldown after any circuit breaker fires
- Pause after 5–7 consecutive losing trades

## Layer 4 — Max drawdown kill switch

- Portfolio down 15–20% from peak equity → shut everything down (with the core plan: the swing sleeve down 15% from its peak stops swing entries until a human /resumes; the core keeps buying and every stop keeps working — Jason, 2026-09-24, D-040)
- Maintain a 20% cash reserve at all times; never fully invested (with the core + swing plan: inside the swing sleeve only, measured on swing-sleeve equity — Jason, 2026-09-24, D-040)

## Layer 5 — Environmental filters

- Reduce size or halt when VIX > 30
- Pause around major macro events: FOMC, CPI, NFP
- Emergency kill switch reachable from mobile: Discord `/killswitch` (stops everything, the core included)
- Auto-cancel all open orders on loss of exchange connectivity

## General rules

- Model 0.1–0.5% slippage in all backtests
- Prefer limit orders over market orders
- High-liquidity instruments only
- Paper trade 1–3 months before any live capital
- Start live with a small amount only
- Exchange API keys: withdrawal permission always disabled
- Complete trade logs retained ≥ 6 years
- Never claim guaranteed returns, anywhere

## Structural enforcement (architecture-level)

- The LLM analyst can only *propose* `{action, symbol, confidence, reasoning}`. The risk gate consumes proposals and applies these layers; the LLM never sees or sets risk parameters, sizes, or stops.
- Kill-switch state lives in Redis and is checked independently by the scheduler, the risk gate, and the executor.
- `TRADING_MODE=paper` is the only mode: this public build is paper-only and rejects `TRADING_MODE=live` outright.
