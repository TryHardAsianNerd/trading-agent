# Spec — what "done" means

What this project is for, what counts as finished, and the bar it must clear before it trades real money. Decided by Jason on 2026-10-18 (D-064). This file wins over older goal statements; change it only with a decision entry.

## Goal

A personal trading agent for Jason that runs unattended on a local machine and, once it has proved itself on paper, **trades live money for him**.

- **70% of the account** sits in index funds and safe investments (the core).
- **30% is the swing budget.** It swing-trades stocks now. Options and shorting come later, inside the same 30% (stage S2 below).
- **No day trading. No leverage. No promised returns**, in either part.

## Two finish lines

1. **Functional done:** the agent does the job unattended. Every statement in F1–F9 and U1–U5 holds.
2. **The live gate:** it has earned the right to touch real money (L1–L6). This public build is paper-only and cannot go live.

## How the money moves

- Each deposit splits 70/30, and the split is recorded when the deposit is booked (D-061).
- **Core, deterministic, no LLM:** VOO 37.5%, QQQM 22.5%, BTC 6%, ETH 4% of the account (D-040). A holding sells in full when a daily close is 25% below its highest close since it was bought (floored at cost), and re-buys after its price retakes the 200-day average.
- **Swing, 13 stocks:** Claude proposes; the deterministic risk gate in `src/trading_agent/risk/` decides. Buy only in an uptrend and only a stock not held (D-038, D-057). Sell only when the trend turns down (D-060) or the stop is hit (D-054). Fractional orders of at least $10, 1% risk per trade, 25% cap per position, 20% cash reserve in the sleeve (D-042). The app holds the stops (D-054).

## F — functional done

Each statement is proven by a test or by a query against real paper data.

| ID | Statement | Status (2026-10-18) |
|---|---|---|
| F1 | Each deposit splits 70/30, and the split is recorded at booking. | Met (live 10-18) |
| F2 | The core buys toward its targets automatically. | Met |
| F3 | A core holding sells on its 25% trail (floored at cost) and re-buys only after retaking its 200-day average. | Met in code; not yet triggered live |
| F4 | Swing buys only in an uptrend and only stocks not held: fractional DAY orders of at least $10, sized by the gate. | Met (10-16) |
| F5 | A swing position exits only when its stop is hit or its trend turns down. | Partial: D-060 live 10-18; no live stop sale yet |
| F6 | Halts work: the kill switch, sleeve breakers and drawdown switch stop new swing entries; stop sales still go out except under the manual kill switch or with Redis down. | Met |
| F7 | Every proposal (with Claude's reasoning), approval, rejection and order is stored and queryable. | Met (live 10-18) |
| F8 | Every Monday the weekly report reaches Discord. It shows each swing trade's result on its own investment, the dollar-weighted swing total against the +5% bar, and, for analysis only, what the same money would have made in VOO. | Report met; the swing-bar and VOO views are to build |
| F9 | Paper is the only mode until the live gate is passed. | Met |

## U — running unattended on a local machine

The app holds the swing stops, so **while the machine is off there is no swing stop at all**. That is the cost of local hosting.

| ID | Statement | Status |
|---|---|---|
| U1 | The agent starts by itself at logon. | Met |
| U2 | Jason is paged on Discord for every error-level event, including the analyst failing 3+ cycles in a row (D-018). | Met |
| U3 | Jason is paged when the agent stops cycling during market hours. **Required before live.** | Not yet |
| U4 | Jason is paged when a swing stop goes unchecked for 3 priced cycles in a row. **Required before live.** | Not yet |
| U5 | The machine is on during NYSE hours on trading days, and at least 95% of market-hours cycles run in each window. | Not enforced during the practice run |

## L — the paper test and the live gate

| ID | Rule |
|---|---|
| L1 | **Window:** 4 weeks of paper trading with no rule changes. The official window starts on the first trading day after Jason says to start it, and the agent records that date here with a decision entry. Tue 2026-10-20 to Mon 2026-11-16 is a practice run under the same rules: a rehearsal of the weekly report and the +5% math, not a verdict. |
| L2 | **Bar:** each swing trade is measured on its own investment, after fees. The test passes when the dollar-weighted total (all swing profit ÷ all money put into swing trades) is **at least +5%** over the window. Positions still open at the end count at market value. A trade that reaches +50% is logged as a big win. VOO is shown in the weekly report for analysis and is not part of the bar. |
| L3 | **No minimum trade count.** If the window has no swing trade at all, there is nothing to measure, and the window is extended by two weeks. |
| L4 | **Guardrails:** zero unexplained trades, zero risk-rule violations, the 15% sleeve breaker never trips, and F1–F9 plus U1–U5 are met. |
| L5 | **First live scope:** stocks only on Alpaca, with a small amount to start. The live core is VOO and QQQM; the core's 10% crypto share goes to them in proportion until crypto rules are tested. Crypto stays on paper. The live checklist is human-only. |
| L6 | **Live review:** every 2 weeks, the last 4 weeks of swing trading are checked against the L2 bar (paper weeks fill in until 4 live weeks exist). A miss sends it back to paper until it passes again. |

## Strategy freeze

- During a test window, no rule changes: fixes only, unless Jason says otherwise (D-062, D-064). A problem found during the window is written down for Jason, not into a rule.
- A rule change restarts the window.

## Later stages

- **S2, options and shorting**, inside the same 30% swing budget: **defined risk only**. That means buying calls and puts, where the most that can be lost is the premium paid; a bet against a stock is a bought put. No selling options, no borrowed shares, no margin. It needs its own design note, risk-engine rules, RISK_RULES edits by Jason, Alpaca options approval and its own paper test. The code today knows only long buys and sells.
- **Crypto rules:** tested separately, later; crypto stays on paper until then.

## Non-goals

- Not a product. No other users.
- No day trading, no leverage, no promised returns.
- No LLM in sizing, limits or the core. Claude only proposes swing trades.

## Known risks

- **Four weeks is a small sample.** Passing means "allowed to start small", not "proven". L6 is the safety net.
- **Day-trade limits on a live account:** an account under $25k may make only 3 same-day round trips in 5 business days, and a stop hit on the buy day counts as one.
- **No stops while the machine is off.** U3 makes an outage visible; only always-on hosting removes it.
