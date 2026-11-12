# Decision Log

Entry numbers have gaps: this public copy leaves out the entries that are private.

---

## D-001 — Follow the blueprint's stack (2026-09-02)

**Decision:** Adopt the trading_agent_blueprint docx as the guiding document: LangGraph orchestration, Alpaca for stocks, Coinbase Advanced Trade for crypto, Polymarket read-only, XGBoost + FinBERT for ML, TimescaleDB/Redis storage.
**Why:** Jason provided it as the product definition; it is internally consistent and current (March 2026).
**Consequences:** The blueprint itself is not part of this repository.

## D-003 — Runs locally via Docker Compose (2026-09-02)

**Decision:** Local machine deployment; no cloud hosting for now.
**Why:** Free, full control, sufficient for paper trading. Everything stays in Compose so a later VPS move is trivial.
**Consequences:** No CI/CD or cloud infra work in Phase 1.

## D-005 — Claude (Anthropic API) powers the analyst node (2026-09-02)

**Decision:** langchain-anthropic / Claude for the LLM analyst.
**Why:** Jason's choice; strong reasoning.
**Consequences:** Requires `ANTHROPIC_API_KEY`; per-cycle LLM cost is a factor to watch.

## D-007 — Single database: TimescaleDB for everything; no SQLite (2026-09-02)

**Decision:** TimescaleDB (which is Postgres) holds both OHLCV hypertables and relational trade/order/portfolio tables. Skip the blueprint's separate PostgreSQL and SQLite.
**Why:** Docker Compose gives us TimescaleDB from day one; one DB is simpler and loses nothing.
**Consequences:** All SQLAlchemy models target one database.

## D-008 — Risk engine is deterministic and LLM-isolated (2026-09-02)

**Decision:** The five risk layers are pure functions in `risk/`; LLM proposals pass through a risk gate that the LLM cannot see, modify, or bypass.
**Why:** Blueprint's "no exceptions and no overrides from strategy logic," made structural.
**Consequences:** `risk/` requires exhaustive unit tests before any executor code merges.

## D-009 — Personal-use scope; productization deferred (2026-09-02)

**Decision:** Build for one user (Jason) on his own accounts. Productization (most likely a signals-subscription first) is a possible future, not a design driver.
**Why:** Auto-trading for others triggers investment-adviser regulation; the personal build produces the track record any product pitch needs anyway. Securities attorney required before any commercialization.
**Consequences:** No multi-tenancy, auth, billing, or key vaulting in the current architecture.

## D-013 — Crypto paper trading runs against an internal paper-sim broker (2026-09-10)

**Decision:** Phases 1–3 crypto "paper trading" uses an internal PaperBroker implementing the Broker protocol, simulating fills against our own market data with modeled slippage (0.1–0.5% per RISK_RULES). The Coinbase Advanced Trade sandbox is NOT the paper venue.
**Why:** The adversarial review established the sandbox serves only Accounts+Orders endpoints (no get_product) and returns static canned data — it cannot exercise fills, stops, or account state truthfully. Jason chose the simulation over sandbox-as-smoke or deferring crypto.
**Consequences:** New issue for PaperBroker; CoinbaseBroker remains the Phase 4 live path and is smoke-tested only until then. The scheduler/executor selects the broker per asset class AND trading mode.

## D-014 — Coinbase entries switch to atomic trigger-bracket orders (2026-09-10)

**Decision:** CoinbaseBroker buys will use trigger_bracket_order_gtc (marketable limit entry + atomically attached stop) instead of market-IOC-then-stop-limit.
**Why:** The adversarial review disproved the "no OTO on Advanced Trade" premise. Bracket eliminates both the unprotected window between fill and stop placement and the fill-read race machinery that guards it. Jason approved the trade-off (limit entry instead of guaranteed market fill).
**Consequences:** ApprovedOrder gains a data-derived reference price (set by the risk gate from its prices input — not LLM-writable) so the broker can price the marketable limit. The terminal-poll machinery stays (still needed to learn fill state) but no longer guards a naked-position window. Built as its own change; the market-IOC implementation merges first as the tested baseline.

## D-015 - Tier-1 indicators are hand-rolled on pandas; no indicator library (2026-09-14)

**Decision:** The feature engine computes the Tier-1 indicator set (log returns, SMA/EMA 10/20/50/200, RSI-14, MACD 12/26/9, Bollinger 20/2, ATR-14, OBV, VWAP-20) as hand-written pandas functions in `src/trading_agent/features/`. No third-party indicator library is added.
**Why:** pandas-ta (the blueprint suggestion) is broken on numpy>=2 and its PyPI release is from 2021; its maintained forks are not pinnable from PyPI (supply-chain risk); `ta` last released in 2023 against older pandas. Tier 1 is eight textbook formulas, each a few lines of pandas, fully verifiable with known-value tests - the same approach already proven for the risk engine's Wilder ATR (risk/stops.py). A dependency bringing 100+ unneeded indicators fails the no-overengineering rule.
**Consequences:** `features/indicators.py` (pure Series functions, float math - features are advisory inputs, not money; Decimal stays mandatory in risk/brokers) + `features/engine.py` (per-symbol latest-row snapshot). ATR parity with risk/stops.wilder_atr is pinned by test so the two implementations cannot drift. Tier-2+ indicators (Phase 2) reuse the same pattern or revisit the library question then.

## D-016 - Fill reconciliation is a per-order venue lookup over our own orders book; halts block breach closes (2026-09-16)

**Decision:** The executor books every venue order id it learns about (entry and protective stop) as an `orders` row and reconciles fills by polling `Broker.get_order(broker_order_id)` on non-terminal rows each cycle — not by listing "recent" venue orders. Brokers own their idempotency (Alpaca replays a rejected duplicate key from the venue). While the kill switch is active the executor submits nothing, including trailing-stop breach closes; while a venue's reconciliation failed, breach closes still go through but new positions do not.
**Why:** Both real venues filter order listings by SUBMISSION time, so a GTC stop placed with an entry weeks ago and filled today falls outside any "recent" window — a point lookup keyed by ids we already hold has no window semantics and one booking path for synchronous (PaperBroker) and asynchronous (Alpaca) fills. Blocking breach closes under a halt matches the gate (which rejects sells under a halt too) and keeps "shut everything down" literal while the venue ATR stop still protects the position; the adversarial review flagged that this disables the tighter 3% exit during exactly the cooldown periods when losses mount, so this half of the decision is Jason's to confirm or reverse. Letting breach closes through a stale book is safe because full closes cannot double-sell at any venue here.
**Consequences:** `Broker` gains `name`, `asset_class`, `get_order`; `SubmittedOrder.protective_order_id`; `UnknownOrderError` for a venue 404 (row retired as `lost`, never blocks the venue). `orders` gets a unique `(broker, broker_order_id)` index; `paper_orders` gets fill columns. New alerts: `position_unprotected`, `submission_unbooked`, `order_lost_at_venue`, `fill_unbookable_no_price`. Reversing the halt half is a one-line change in `trade_executor.execute` plus its test.

## D-017 - Discord replaces Telegram as the human interface (2026-09-20)

**Decision:** The Phase 1 notification and control channel is a Discord bot (discord.py) in one private server/channel, not a Telegram bot. Supersedes the earlier Telegram plan; "no web UI, chat bot + logs" stands.
**Why:** Jason's preference (2026-09-20). Functionally equivalent for a single-user tool: free, phone push, one bot, one allowed user id for commands. The notifier is one module, so the choice is cheap to make and cheap to reverse.
**Consequences:** `notify/discord.py` instead of `notify/telegram.py`; env vars `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`, `DISCORD_OWNER_ID` replace the Telegram ones; `python-telegram-bot` is dropped from dependencies in favour of `discord.py` (pinned). Commands are `/killswitch`, `/resume`, `/status`, accepted only from the owner id. RISK_RULES.md Layer 5 names Telegram for the mobile kill switch; that file is human-only — Jason to update the word, the rule is unchanged.

## D-018 - Log level is an alerting contract; Discord messages carry type names and ids, never exception text (2026-09-20)

**Decision:** `log.error`/`log.exception` PAGES A HUMAN: the notifier drains our error-level structlog events into the cycle's Discord alert. Per-symbol degradations the cycle absorbs by design (a failed quote, a provider 429 or plan restriction, a per-symbol feature failure) are WARNINGS. Third-party (stdlib) log records are never collected. Every Discord message — alerts, crash notices, command replies — carries exception TYPE names and structured ids only (event, symbol, broker, order ids, reason, count); raw exception text (`error`/`detail` fields, BrokerError messages) stays in the JSON logs and is never rendered.
**Why:** With alerts wired to the log stream, the level of a log call is a paging decision, so the meaning of "error" has to be fixed once or the channel becomes noise (per-symbol warnings every cycle) or silence (a real naked position downgraded to a warning). Exception text from venue SDKs and HTTP clients embeds URLs, response bodies and occasionally credentials; a code review found `UnprotectedPositionError` and the executor's `error`/`detail` fields forwarding raw Coinbase SDK text into the alert, and "kept safe by the SDK pins" is not a policy that survives a pin bump. Type names plus our own ids are enough to know what to look at; the logs hold the rest.
**Consequences:** `log.collect_alerts` collects only structlog-originated error/critical events; `notify/discord.py` renders `_ALERT_FIELDS` without `error`/`detail` and result lines without `detail`. New code that logs at error level is declaring the event page-worthy; anything absorbed by the cycle logs at warning. Adding an alert field that can hold library text is a policy violation, not a pin question.

## D-019 - Connectivity watchdog pauses new entries and never cancels protective stops; halt semantics per reason (2026-09-21)

**Decision:** (1) When a cycle cannot trust its view of a venue — the portfolio could not be assembled, venue order lookups failed, or the cycle crashed — the scheduler arms the one-hour breaker cooldown (auto-releasing; a persistent outage re-pauses and re-pages hourly) and pages Discord. It never cancels open orders. This supersedes the literal RISK_RULES Layer-5 wording "auto-cancel all open orders on loss of exchange connectivity" for this system. (2) Halt reasons are handled differently by the scheduler: the kill switch and an unreachable Redis skip the whole cycle (no LLM/provider spend while a human holds the switch); a gate breaker cooldown still runs the cycle (the gate rejects and the executor refuses, but snapshots, fill reconciliation and simulated-stop settlement continue); the scheduler's own pauses skip the cycle and post no further notices. (3) A restart clears the scheduler's own pauses (the restart is the human act); gate breaker cooldowns are never cleared by the scheduler. (4) A crashed cycle is resumed once under the same cycle_id; a second crash abandons it. (5) With Discord configured, no cycle runs until the bot's gateway reports ready: a gateway that never comes up keeps the agent waiting (warning every 30s) rather than trading without the phone kill switch.
**Why:** The only open orders this system holds are the protective stops attached to each position; cancelling them on a connectivity blip would strip every position of its protection, and if the connection is really gone the cancel cannot be sent anyway. Pausing new entries fails closed without removing protection; the human keeps `/killswitch` for "cancel everything". Running cycles through gate cooldowns preserves the book (venue-side stop fills, snapshots for the day/week anchors) and avoids hourly HALTED/cleared chatter when a bad day re-trips the daily breaker. Resuming a crashed cycle replays only the unfinished nodes with the same idempotency keys, so an order submitted just before the crash is booked instead of orphaned. Jason chose the watchdog semantics on 2026-09-21; the adversarial review of the change shaped (2)-(4).
**Consequences:** `scheduler.py` (AgentRunner, serve); `KillSwitch.clear_cooldown(*reasons)`; `breaker_cooldown_started` logs at error level (it pages, D-018); `AgentState.portfolio_error` / `AgentState.reconcile_failed` are the watchdog's inputs (type/venue names only, never exception text). RISK_RULES.md is human-only: Jason to reword Layer 5's connectivity line to match if he wishes; the rule's intent (fail closed on connectivity loss) holds. Known gap: the resume-once id lives in process memory, so a process death between a venue submission and its booking is not resumed on restart — the position surfaces as `untracked` at the next cycle (the gate blocks adds to it) instead of being booked under its idempotency key.

## D-020 - The kill switch means "cancel everywhere", in the executor too; cooldowns never cancel (2026-09-21)

**Decision:** When the executor observes a `kill_switch:*` halt it calls `cancel_all_orders()` on every configured broker before reconciling and audits the per-venue counts (or the failure's exception TYPE name) under `cancelled` in its decisions row; a failed venue cancel logs at error level (pages, D-018) and never stops the other venues or the cycle. A `cooldown:*` halt and `redis_unreachable` never cancel — the executor only refuses to submit, exactly as before. Nothing in the executor clears the kill switch. **Scope (Jason, 2026-09-22, option B):** only the HUMAN kill switch (`kill_switch:manual_discord`, the reason `/killswitch` stores — `MANUAL_KILL_SWITCH_REASON`) triggers the executor's cancel-everywhere. The gate's automatic `max_drawdown_exceeded` kill switch halts new entries and skips cycles exactly as before but leaves the venue-side ATR stops working: stripping protection during the period a drawdown breaker fires is the worst of both worlds, and the human path still gets the full cancel. If a drawdown trip should actively CLOSE positions, that is a separate decision.
**Why:** `/killswitch` arms the flag and cancels, but a cycle already past the executor's halt check can still submit afterwards (and `--once` runs have no scheduler skip); until now those orders survived until the human re-ran the command. Cancelling in the executor closes that race idempotently (a venue with nothing open reports 0) and keeps the switch literal, matching `_cancel_everywhere` in the bot. Cooldowns and Redis outages are D-019 territory: the only open orders this system holds are protective stops, so cancelling on a breaker pause or a lost safety-flag read would strip protection during exactly the periods it matters. Cancel runs BEFORE reconciliation so the same cycle's poll retires the cancelled rows and so a DB outage in reconcile cannot keep the kill switch from reaching the venues.
**Consequences:** `trade_executor._cancel_everywhere`; decisions rows for node `trade_executor` carry `cancelled` (None unless the halt is the kill switch). The bot's "re-run /killswitch if a cycle was in flight" advice is now belt-and-braces rather than the only fix. D-016's halt semantics (no submissions under any halt, including breach closes) are unchanged.
## D-021 - Alpaca IEX daily bars are the stock candle source; Finnhub keeps quotes (2026-09-22)

**Decision:** Stock daily OHLCV comes from Alpaca's market-data API (`StockHistoricalDataClient`, the same paper keys as `AlpacaBroker`), IEX feed, split-adjusted, normalized to the repo's 00:00 UTC bar-open convention, `source="alpaca"`. Ingest routes stock candles to Alpaca whenever the Alpaca keys are configured and falls back to the Finnhub candle path only when they are not. Stocks re-fetch the full 365-day window every cycle instead of resuming from the latest stored bar. Finnhub stays the stock QUOTE source. SIP is a paid feed and is never requested.
**Why:** Finnhub's free plan 403s on `/stock/candle`, so every stock proposal was rejected INSUFFICIENT_DATA (the ATR stop needs 15+ bars; the long moving averages want 200). The Alpaca keys already exist, the SDK is already pinned, and one request returns the whole 365-day backfill (~250 bars) — no new vendor, key, or dependency. Split adjustment keeps a split from faking an ATR spike while leaving the recent closes equal to venue prices (dividend adjustment would move them); re-fetching the whole window each cycle costs the same one request as a cursor fetch (~250 bars, one page) and lets the upsert rewrite pre-split rows the cycle after a split, whereas a cursor would leave them inflating the ATR and skewing every moving average for up to 200 sessions with nothing to detect it. Tiingo remains the option if Phase 3 needs deeper history or consolidated volume.
**Consequences:** `data/providers/alpaca.py` (`AlpacaDataClient.get_daily_candles`, errors sanitized to type name + HTTP status — alpaca-py's `APIError` text is the raw response body); `CycleDeps.alpaca_data`; `ingest_daily_ohlcv(alpaca=...)`. IEX bars carry IEX-executed volume only (a few percent of consolidated), so OBV/VWAP features see a partial tape; the risk engine reads prices only. Old finnhub rows would coexist harmlessly (none exist in production; `recent_daily_candles` dedupes by timestamp with a deterministic source tiebreak). The `latest_ts` cursor now serves crypto only. `tzdata` is declared (Windows has no system tz database).

## D-022 - Liquidity filter: 20-day average dollar volume, measured on the IEX tape (2026-09-22)

**Decision:** The risk gate enforces RISK_RULES' "high-liquidity instruments only" as a sixth per-proposal check (`risk/liquidity.py`). A BUY is rejected `RejectReason.ILLIQUID` when the average daily DOLLAR volume over the last `liquidity_window_days = 20` distinct-timestamp daily bars is below the asset class's frozen minimum: **$10,000,000/day for stocks** (measured on the Alpaca IEX tape, see below) and **$50,000,000/day for crypto**. Both live in `RiskLimits` as `Decimal` constants — never env vars, never config, never anything LLM output can reach (D-008). SELLS are never liquidity-filtered: closing a position reduces risk, and thin volume must not trap us in a name. Fewer than 20 distinct-timestamp bars is INSUFFICIENT_DATA, the path that already covers a too-short ATR history, not ILLIQUID. The check runs BEFORE the ATR/stop/sizing work, so a rejected symbol consumes no batch cash and no position-cap headroom. Rejections are audited under `audit["illiquid"]` with the measured figure. Only VOLUME is enforced; the spread half of the rule is deferred (see Consequences).
**Why:** The rule was documented and unenforced (flagged in the S3 adversarial review); nothing stopped the LLM proposing a micro cap. Dollar volume, not share volume, is the size-independent measure and the only one the stored data supports. 20 sessions is about one trading month — long enough that a single quiet day cannot flip a verdict, short enough to react to a name going quiet, and within CoinGecko's 30-day crypto backfill cap.
**Feed investigation (live, 2026-09-22).** Alpaca's `DELAYED_SIP` feed returns HTTP 400 on the paper key. Plain `SIP` DOES return consolidated bars on this plan, but only when the request's `end` is at least ~15 minutes in the past; ingest asks for `end=now` and gets HTTP 403. Switching stock bars to consolidated volume would therefore mean reversing D-021's "SIP is never requested", changing the ingest window, and moving every stock OHLCV row the system stores — far beyond a liquidity filter, and a decision for Jason, so **the IEX feed is unchanged and the stock threshold is calibrated ON THE IEX TAPE.**
**Threshold calibration (live 20d IEX dollar volume, 2026-09-22):** SPY $964M, NVDA $729M, AAPL $399M, MSFT $298M, KO $73M, F $24.9M, CAG $8.05M, SIF $0.04M, CULP $0.01M, LGL $0.004M. Measured IEX share of the consolidated tape on the same names: 2.5-5%, so $10M IEX is roughly $250-400M consolidated. The configured watchlist's least liquid name (MSFT) clears it by ~30x while a micro cap misses by 250-1000x; names at the illiquid bottom of the S&P (CAG-class) fall just under, which is the intent — the filter admits mega and large caps only. Conservative in the direction that costs opportunity, never money, matching the rest of `limits.py`. Crypto: BTC/ETH run in the tens of billions per day (measured from stored candles: BTC $27.6B, ETH $13.1B), so $50M only excludes a dead market.
**Volume units differ by asset class — the load-bearing subtlety.** Alpaca stock bars carry a SHARE COUNT, so a bar's dollar volume is `close * volume`. CoinGecko's `market_chart.total_volumes` is denominated in the vs_currency, so a crypto bar's stored `volume` IS the dollar figure already (a rolling-24h approximation). Multiplying the crypto value by the close would overstate it by ~5 orders of magnitude and make the filter a no-op for crypto. The unit rule is decided once from the asset class (itself derived from the candles, never from the proposal) and pinned by test.
**Consequences:** `risk/liquidity.py` (`average_dollar_volume`, `is_liquid` — pure, Decimal-only, de-duplicating by timestamp exactly as `wilder_atr` does); `RejectReason.ILLIQUID`; three new `RiskLimits` constants; `gate._evaluate_one` gains the check on the buy path. The IEX caveat means the stock number is NOT comparable to a published ADV — if stock bars ever move to a consolidated source, `min_avg_dollar_volume_stock` must be re-calibrated in the same commit or the filter silently becomes ~30x looser. The concrete path is `data/ingest.py`'s Finnhub candle fallback when no Alpaca keys are configured (consolidated volume, and a raw epoch `ts` that would not dedupe against Alpaca's NY-midnight normalization): unreachable today (free-plan 403, and no finnhub rows exist), and deliberately NOT guarded in `risk/` — pinning a provider name inside the risk engine would couple it to `data/providers/` and break the documented Tiingo path, while the exposure is a looser filter (down to ~$10M/day consolidated), not a money-losing bug at this account size. Adding a source guard is Jason's call. A partial bar for an open session lowers the average and biases toward rejecting, which is the safe direction and is not special-cased. **Spread is not enforced:** nothing in this system stores a bid/ask — `data.models.Quote` carries a single price — so a spread filter needs a new data path; dollar volume is the standard proxy and is what the stored data supports. Revisit if a quote source with a book lands.

## D-023 - Layer 5 reads FRED's VIX close and a checked-in macro calendar; a degraded environment is benign-but-loud (2026-09-22)

**Decision:** `MarketEnvironment` is built once per cycle in the risk node. VIX is the latest non-missing observation of FRED series `VIXCLS` (the daily CLOSE, requested newest-first, `"."` placeholder rows skipped), so the gate's "halve buys above 30" rule always acts on the most recent published close — typically one trading day old, and older across weekends and holidays. A fetch failure or an unconfigured `FRED_API_KEY` yields `vix=None` with a WARNING, never an error and never a halted cycle. A close older than `RiskLimits.vix_max_age_days` (7) warns but is still used. The macro pause comes from `risk/macro_calendar.json`, a checked-in list of scheduled FOMC statement (14:00 ET, second day of each two-day meeting), CPI and Employment Situation (08:30 ET) releases with their source URLs cited in the file; `macro_pause()` is a pure function of `(now, events, limits)` that returns True inside `[30 min before, 60 min after]` each release, converting America/New_York clock times with zoneinfo so DST is the tz database's problem. Both inputs are recorded in the decisions row under `audit.environment` (`vix`, `vix_date`, `macro_event_pause`).
**Why:** VIX is a SIZE REDUCER, not a safety stop — halving buys is a comfort measure, so an unavailable VIX degrading to the pre-existing benign default ("unknown, no halving") is strictly better than pausing trading on a data-vendor outage. The 7-day cap WARNS but does not discard, which the adversarial review of this PR corrected: the gate has exactly two states (halve, don't), and a stale CALM reading is already behaviourally identical to `None`, so dropping a stale value could only ever be the transition halve -> don't-halve. Discarding it would mean a series that froze during a genuine vol spike quietly resumed full-size buys on day 8 — the cap as originally written could only ever make the system less conservative, which is the wrong direction for a risk limit (RISK_RULES: take the conservative end). The pause window is asymmetric because the hazards are: 30 minutes before, market makers widen and pull quotes into the print; 60 minutes after covers the initial repricing, the first half hour of the 14:30 FOMC press conference, and the 09:30 cash open following an 08:30 BLS release. A checked-in calendar beats a scraped one — these dates are published a year ahead, change rarely, and a parser failing at 08:29 on CPI day is a worse outcome than editing a JSON file each autumn. **The key detail is secret hygiene: FRED authenticates with the API key AS A URL QUERY PARAMETER and offers no header option, so every request URL is a secret.** Sanitizing our own exceptions (type name / status only, `from None` to cut the `__cause__`/`__context__` chain structlog renders) is only half the surface: httpx logs `HTTP Request: GET <full url>` at INFO on the SUCCESS path, and `log.configure_logging` deliberately bridges third-party stdlib records into the same JSON stream — so the live key would have been written to the logs on every cycle. Caught while reviewing the draft of this change and reproduced with a fake key before it was fixed — the WIP's exception sanitization was already careful, which is exactly why the success-path log line was easy to miss.
**Consequences:** `data/providers/fred.py` (`FredClient.latest_vix`, 20s read timeout — the observations endpoint measured 6-10s live, and the 10s used elsewhere timed out more often than not, which would have left VIX permanently "unavailable"); `risk/macro_calendar.py` + `macro_calendar.json` (shipped as package data); `RiskLimits.macro_pause_before_minutes` / `macro_pause_after_minutes` / `vix_max_age_days`; `CycleDeps.fred`; `FRED_API_KEY`. Secret hygiene is now TWO layers, because the review showed one was not enough: (1) `log.configure_logging` pins the `httpx` and `httpcore` loggers to WARNING, guarded by `test_configure_logging_pins_the_noisy_http_loggers`; (2) `log.redact_url_secrets` is a structlog processor that rewrites any credential-shaped query param (`api_key`, `x_cg_pro_api_key`, `token`, `secret`, …) to `REDACTED` at the render choke point, descending into dicts/lists and covering non-string values via the repr JSONRenderer would emit, and running both in the shared chain (before `collect_alerts`, so Discord sees the redacted value) and again after `format_exc_info` (where a traceback first becomes a string). **Each layer needs its own test, because the end-to-end test passes while EITHER holds:** the second review of this PR deleted the pin and watched the whole suite stay green, since redaction silently covered the same line. Two layers that collapse into one without anything going red are one layer. Layer 1 alone was a two-name log-LEVEL whitelist: a record at WARNING on `httpx`, or an INFO record from any other library quoting the URL, still rendered the key — all three demonstrated in review. `httpx` is pinned `>=0.28,<0.29` for the same reason `alpaca-py` is: its logging behaviour is a security property of this system, and an unbounded range means a routine upgrade could promote the request line to WARNING with nothing in CI noticing. **Maintenance: the calendar runs out on 2026-12-04 (the last NFP; CPI runs to 2026-12-10, FOMC to 2027-12-08), so the tripwire fires 2026-11-04/05 — exactly 30 days ahead, verified.** BLS publishes the next year each autumn; an exhausted calendar is indistinguishable from a calm market at the call site, so the node logs `macro_calendar_stale` every cycle once any kind runs out and `test_shipped_calendar_is_not_about_to_run_out` goes red 30 days ahead — measured against `datetime.now`, because the hardcoded-date version this PR first shipped would have kept passing forever after the calendar expired. Layer 5's remaining literal RISK_RULES line (auto-cancel on connectivity loss) is still superseded by D-019.


## D-024 - XGBoost directional model: Alpaca history, price-only features, purged walk-forward, skill-floored and advisory (2026-09-23)

**Decision:** The `ml_predictor` node serves a pooled XGBoost classifier of P(close[t+5 bars] > close[t]) for every priced symbol. (1) **Training is an offline CLI** (`python -m trading_agent.ml.train`) — no scheduler hook until the Celery worker exists. It fetches years of daily bars straight from Alpaca and never writes them to the `ohlcv` table: stocks through the same IEX/split-adjusted client the cycle already uses (D-021; history from mid-2020), crypto through Alpaca's crypto bars (from 2021, 00:00 UTC stamps, symbol `BTC-USD` requested as `BTC/USD`). (2) **Features are 14 scale-invariant, PRICE-ONLY values** (log returns over 1/5/10/20 bars, 10/20-bar return volatility, RSI/100, distance from SMA-10/20, Bollinger %b, ATR/close, MACD-hist/close, bar range/close, an is_crypto flag) computed by ONE function (`ml/dataset.feature_frame`) for training and inference; the longest lookback is MACD's 34 bars. (3) **Evaluation is a purged expanding-window walk-forward**: the second half of the timeline is cut into 5 consecutive test blocks, and each block's model trains only on rows whose label window closes before the block opens. (4) **Artifacts** are XGBoost's native JSON plus `meta.json` (features, horizon, params, walk-forward metrics, data sha256, xgboost version) under `models/directional/<UTC timestamp>/` (gitignored), written to a hidden staging dir and renamed into place. The cycle loads the newest version-named directory every cycle and refuses one whose recorded features/horizon differ from the code's (no fallback to an older version). (5) **Skill floor:** a version is served only if BOTH its pooled walk-forward AUC and its median fold AUC exceed `MIN_OOS_AUC = 0.52` (`model.is_servable`). (6) **Inference uses complete bars only** (bars dated before the cycle's UTC day), needs at least 60 of them, and skips a symbol whose newest complete bar is older than 1 day (crypto) / 4 days (stocks). (7) Predictions reach the **LLM analyst prompt only**, as `{"model": {version, horizon_bars, oos_auc, base_rate}, "by_symbol": {symbol: {prob_up, as_of}}}`, and the system prompt tells Claude to judge `prob_up` against `base_rate`, not 0.5.
**Why:** The stored data cannot train anything: crypto in the DB is a 30-day backfill plus what the cycle has appended since, and a model trained on the same DB rows it scores has no honest out-of-sample test. Alpaca's history is free with the keys we already hold (no new vendor, so no escalation), and for stocks it is literally the serving feed. Volume features are excluded because the two sides measure different things — crypto trains on Alpaca's single-venue volume but serves on CoinGecko's rolling-24h aggregate, and IEX stock volume is a partial tape — so a volume feature would train on one quantity and serve on another. Scale-invariance lets one model pool symbols from ~$20 to ~$90,000, which is the only way to get ~10k training rows from a 6-symbol watchlist. K-fold on overlapping 5-bar labels leaks the future into training; the purge closes it, and the walk-forward and the training job share the same `fit`. The 60-bar minimum is a train/serve-skew guard found in review: training rows sit on years of warm EMA/Wilder state, and at 34 bars BTC's MACD-hist feature measured ~50% off its full-history value (converging by ~60), so crypto (DB backfill started 2026-09-04) gets NO predictions until it has 60 complete bars — early October 2026. The stale-bar limit stops a days-old prob_up from being presented as current when ingest has been failing. The median-fold condition exists because pooling scores from five fold models can credit calibration shifts as ranking skill and one lucky fold can carry the pool. The first live training run (2026-09-23): 10,143 rows, walk-forward **AUC 0.5296**, accuracy 0.5363 against a **base rate of 0.5571** (always-predicting-up would have scored higher accuracy), fold AUCs 0.53/0.50/0.56/0.56/0.50 (median 0.53). That is weak ranking skill at best — the overlapping labels mean the effective sample is ~1/5 of the nominal one — so the floor exists to stop a coin-flip model from being dressed up as a signal for the LLM, and the metrics travel with every prediction so the analyst can discount it. The model is advisory by construction: the risk gate never reads `predictions` (D-008), so a bad model can at worst move the LLM's proposals, which still pass every deterministic layer.
**Consequences:** `ml/dataset.py`, `ml/model.py`, `ml/train.py`; `AlpacaDataClient.get_crypto_daily_candles` (shared sanitized-error path with the stock method); `CycleDeps.model_dir`; a symbol-level failure in the node is a warning and costs only that symbol; `xgboost` pinned `>=3.4,<4` (scipy arrives transitively). A missing, incompatible, or below-floor model logs a warning every cycle and serves `{}`; a DB failure during inference logs at error level (pages, D-018) like the feature engine. **Retraining is manual** until the worker schedules it; nothing warns when a model ages. Changing `FEATURES` or `HORIZON_BARS` makes every stored model unloadable until the job is re-run — by design. **Known residual skew:** the review measured CoinGecko-derived daily crypto closes 0.4% (BTC) to ~1% (DOGE) off Alpaca's 00:00 UTC closes, consistent with CoinGecko 4h stamps being candle CLOSE times so each stored day is shifted 4h; this predates the model, also touches the risk gate's crypto ATR, and needs its own end-to-end reproduction before a fix (the alternative for the model is serving crypto features from Alpaca bars). Evaluation metrics are recorded, not yet tracked over time (Phase 3's decision/outcome analytics should score live predictions against realized moves).

## D-025 - Celery worker in its own container for background jobs; weekly model retrain (2026-09-23)

**Decision:** Background jobs run on a Celery worker (`src/trading_agent/worker.py`) in a `worker` service in docker-compose, built from a new worker-only `Dockerfile`. The trading agent itself keeps running on the host with APScheduler (D-003 unchanged). Worker and beat share one container (`--beat --concurrency=1`, non-root). The broker is Redis database 1 (`CELERY_BROKER_URL`), separate from the kill-switch keys in database 0; results are not stored. The first job retrains the directional model every Sunday 06:00 UTC into the bind-mounted `./models`, and posts a plain-language summary to Discord (skill, accuracy vs always-guessing-up, whether the agent will use it) — or `model retrain FAILED: <exception type>` on failure. Jason chose worker-first (before FinBERT) and Alpaca's News API for FinBERT on 2026-09-23.
**Why:** Celery was deferred until something background-heavy existed; model retraining is that, and FinBERT (torch + transformers, ~1 GB) is next — the container is where those heavy libraries will live, keeping the host install light. A container rather than a host process because Celery's default worker pool does not run on Windows, and a Linux image gives the heavy ML stack one known platform. One container with beat embedded is enough for one weekly job; a separate beat service would be a second moving part for no present need. Sunday 06:00 UTC: both markets' weekend, well away from any cycle doing real work. Discord is the report channel because a scheduled job that fails silently is how a model quietly ages — the problem D-024 left open.
**Consequences:** `Dockerfile`, `.dockerignore` (keeps `.env*` out of the image; keys arrive at runtime via `env_file`), `worker.py`, `ml.train.train_from_alpaca` (shared by CLI and worker; `train()` now returns the metadata), `Settings.celery_broker_url`, `celery[redis]` pinned `>=5.6,<5.7`. Kombu pins the `redis` client below 6.5, so the lock moves 8.1.0 -> 6.4.0 (API used by the kill switch unchanged). The worker image holds a COPY of the code: code changes need `docker compose up -d --build worker`. XGBoost training differs slightly across platforms, so worker-trained and host-trained versions from the same data are not byte-identical. From the review: a 25-minute SOFT time limit raises inside the job so a hang (alpaca-py sets no request timeout) is reported on Discord before the 30-minute hard kill, which would otherwise end it silently; `.env` is optional for compose (`required: false`) so infra-only startup still works without one; the container receives the WHOLE `.env` (every key, not just Alpaca + Discord) — accepted, since it is the same machine and user as the host agent, which already holds them all, and a hand-maintained per-variable list would drift as jobs are added; and because a skipped run (the machine off on a Sunday morning right after a rebuild — the beat schedule lives inside the container) would otherwise go unnoticed, the predictor now warns `ml_model_stale` every cycle once the served model is more than 8 days old (still served: an old model is no worse than it was).

## D-026 - FinBERT scores Alpaca news headlines on the worker; 24-hour sentiment joins the analyst's features (2026-09-23)

**Decision:** (1) **Source:** Alpaca's News API (Benzinga headlines; free with the Alpaca keys already held; stocks and crypto) — Jason's choice 2026-09-23 over Finnhub company news (stocks only). Crypto is requested as `BTCUSD` and mapped back to the repo's `BTC-USD`; each article keeps only the watchlist symbols it mentions. (2) **Scoring:** a worker job every 10 minutes fetches the last 24 hours, skips article ids already stored, and scores only the HEADLINE with `ProsusAI/finbert` pinned to commit `4556d13015211d73dccd3fdd39d39232506f3e43`; rows go to a new `news_sentiment` table (positive/negative/neutral probabilities, model revision). (3) **Heavy dependencies stay off the host:** torch (CPU build) and transformers are an optional `sentiment` extra installed only in the worker image, pinned by `requirements-sentiment.lock`; `ml/sentiment.py` imports them inside `FinBert` only. (4) **Consumption:** the feature engine adds `news_sentiment_24h` (mean positive-minus-negative over the last 24h, -1..1) and `news_count_24h` per symbol, present only when at least one headline exists; the analyst prompt explains both. The directional model does not use them (it would need years of scored history it does not have).
**Why:** Headlines, not summaries: FinBERT was fine-tuned on single financial sentences, and Benzinga summaries run long and blend stories. Re-reading a 24-hour window and skipping stored ids needs no cursor and heals any missed run. The model repo ships only a pickle-format `pytorch_model.bin`; transformers loads it with torch's weights-only loader (no code execution), and pinning a commit means a later upstream change cannot swap the weights under us. The repo's open community "safetensors" conversions were considered and rejected — they are unmerged pull requests anyone can open. Absent-when-unknown matches the feature engine's existing convention (D-015): a neutral 0 for a symbol with no scored news would be indistinguishable from a dead pipeline. Failed runs warn instead of paging because the next run 10 minutes later retries the same window; a genuinely stopped pipeline surfaces as `news_sentiment_stale` on the agent side.
**Consequences:** `AlpacaDataClient.get_news` (sanitized errors like the bar methods), `data.models.NewsArticle`, `storage.models.NewsSentiment` + `NewsSentimentRepository`, `ml/sentiment.py`, the `score_news` task, the analyst system prompt. The worker now needs the database (`DATABASE_URL` pointed at the `timescaledb` service; it runs `init_db` itself on first use) and an `hf_cache` volume for the ~440 MB weights. Worker image grows to ~3 GB. Articles tagged with many symbols (market wraps) count toward each — the prompt tells Claude a diluted, thin signal is thin. Real-model tests (`tests/test_finbert_model.py`) skip on the host and run inside the image. From the review: (a) transformers 5.x, on every `.bin` load, starts a background thread that asks a third-party Hugging Face Space to convert the model and DOWNLOADS the repo's open community safetensors PR into the cache (observed: `refs/pr/29`, ~440 MB) — not loaded, since the pin is not `main`, but unvetted weights on disk and third-party calls on every worker start; the image now sets `DISABLE_SAFETENSORS_CONVERSION=1` (plus quiet progress bars/verbosity for the JSON logs) and the cache volume was wiped and re-downloaded clean. (b) The sentiment join runs in its OWN session and try-block after the price features: a failure there (missing table, a transaction aborted by an earlier failed candle read) warns `news_sentiment_unavailable` and keeps every price feature. (c) Staleness is a heartbeat, not the newest headline: the worker upserts `job_runs(score_news)` after every successful run (even with no new headlines), and the agent warns `news_sentiment_stale` once it is over 1 hour old — the old 24-hour newest-headline check let a dead worker feed a shrinking window for most of a day.

## D-027 - Crypto Fear & Greed joins crypto features; Quiver is not built (2026-09-23)

**Decision:** The feature engine fetches the latest Crypto Fear & Greed Index from api.alternative.me (free, no key) once per cycle — only when a crypto symbol is priced — and adds it as `crypto_fear_greed` (0-100, a float) to every crypto symbol's features; stock symbols never carry it. A reading dated more than one day before the cycle's UTC date is dropped (`fear_greed_stale`); a failed fetch is `fear_greed_unavailable`; both are warnings and cost nothing else. The analyst prompt explains it as market-wide crowd mood. Quiver Quantitative (congressional trades, the other half of the same task) is NOT built: its API is paid, and Jason declined it on 2026-09-23.
**Why:** The index is one market-wide number, so it rides inside the existing per-symbol `features` rather than a new top-level prompt key — the prompt's four-key whitelist test exists to keep non-market data out, and extending it for one number would weaken that tripwire. It is crypto-only in construction (volatility, volume, social, dominance, trends of crypto), so putting it on stocks would mislabel it. The value updates once a day at 00:00 UTC, so yesterday's reading is normal until the publish lands; older means the source stalled. No storage: the API serves its own full history (`limit=0`) should a model ever train on it. Quiver: politicians' trades are infrequent for a four-mega-cap watchlist and disclosed weeks late; not worth a subscription now.
**Consequences:** `data/providers/fear_greed.py` (`FearGreedClient.latest`, sanitized errors), `CycleDeps.fear_greed` (wired in `make_deps`), `feature_engine._add_fear_greed`, one prompt line. The Quiver half is dropped, not deferred. From the review: a future-dated reading is rejected like a stale one (bad timestamp or clock skew); httpx's timeout is per connect/read, not a total deadline, so a server trickling bytes can hold a cycle longer than 10s — accepted, matching every other provider here, and absorbed by a 15-minute cycle.

## D-029 - Backtests replay the model through the live risk gate, day by day; not VectorBT (2026-09-23)

**Decision:** The backtest harness is a daily event-driven simulator (`backtest/engine.py`) that calls `risk.gate.evaluate` — the function the live cycle calls — with the live `RiskLimits`, instead of a VectorBT vectorized simulation. The signal is the directional model's OUT-OF-SAMPLE walk-forward score (`ml.train.walk_forward_scores`, the same purged folds the training job reports): buy when flat and score >= base rate + edge, full close when held and score <= base rate - edge, where the base rate comes from training rows only. Orders fill at the next open, stops (max of the ATR stop and the 3% trail) at the stop or the gapped open, slippage against us on every fill, at both ends of RISK_RULES' 0.1-0.5% range by default. Cooldowns and the kill switch run on an in-memory Redis stand-in with a simulated clock; `gate.evaluate` gains an optional `now` so the weekly breaker's rest-of-week pause is measured from simulated time (default None keeps the live behaviour). The benchmark is equal-weight buy-and-hold over the same window.
**Why:** The question a backtest must answer here is what the RISK RULES do to a signal — sizing at 1% risk, a 5% cap, a 20% reserve, breakers that stop trading after losses. Those are portfolio-level and path-dependent; a vectorized engine would need them re-implemented, which means a second copy of the risk engine that can drift from the one that trades. Calling the live gate makes that impossible. VectorBT's advantage is speed on large parameter sweeps; six symbols over ~1,100 days run in about a second per configuration, so it would buy nothing and add numba/numpy pin friction. The LLM analyst is not backtested: it has read the history it would be trading (lookahead by construction) and replaying it daily for three years would cost real money for a meaningless number — the model's scores are the honest, deterministic proxy. Using walk-forward scores means no day is traded on a prediction trained with knowledge of that day.
**Consequences:** `backtest/engine.py`, `backtest/metrics.py`, `backtest/run.py` (CLI, report to gitignored `backtests/`); `ml.train.walk_forward_scores` (walk_forward now derives its metrics from it); `gate.evaluate(..., now=None)` (risk/ change: one optional parameter, tested). The crypto liquidity floor is zeroed in backtests only (`backtest_limits`), because Alpaca's single-venue crypto volume is not the CoinGecko aggregate that floor is calibrated on; every other limit is live. The adversarial review shaped the reporting: (a) crypto fills pay Coinbase Advanced's ~0.6% entry-tier taker fee per side on top of slippage (stocks: Alpaca, commission-free) — the paper venue does not (fixed in D-030); (b) each walk-forward fold's signal is judged against ITS OWN training base rate (known at fold time), not fold 0's; (c) returns are measured from the starting cash, and buy-and-hold holds a symbol's share in cash until its first bar (it had been valuing a Saturday start at Monday's close); (d) every run reports its mean exposure and buy-and-hold scaled to that exposure, because the 5% position cap limits six symbols to ~31% invested and the fully invested benchmark says nothing about skill; (e) the loss-streak window is the tracker's constant — the day/week anchors and streak remain a parallel implementation of `PortfolioTracker`'s (it needs the database), kept deliberately small. Not simulated: Layer 5 (no VIX series, no macro calendar before 2026), intraday breaker/drawdown checks (daily closes only, so live trips can come sooner), partial fills. Selection bias: the model, its features and the edge grid were all chosen with this window visible. **First result (2026-09-23): every configuration lost money** (-6% to -16%, the worse ones tripping the drawdown kill switch) against +14-16% for buy-and-hold at the same average exposure — the walk-forward model's weak signal (AUC 0.53) does not survive the stops, sizing and costs. That is a finding for Phase 3 (stops churn on daily bars; the loss-streak breaker blocks many signals; the model may simply have no tradable edge), not a verdict on the live LLM-driven system, which cannot be backtested. Tuning the limits themselves stays human-only (RISK_RULES).
## D-030 - The goal is staged profitability after fees; the paper venue charges Coinbase's taker fee (2026-09-23)

**Decision:** (1) **Goal.** Jason's objective for the app is personal profit, pursued in stages: first be profitable after fees in the backtest and beat buy-and-hold at the same average exposure; then hold that through the 8-week paper-trading period; then go live with a small amount. Tuning decisions (stops, sizing, signal) are judged against that, on data they were not tuned on. A +5-10%-per-week target was discussed and set aside: compounded it is roughly +1,100% to +14,000% a year, far beyond what any sustained strategy delivers, and a strategy volatile enough to aim for it would trip the 5% weekly pause and the 15% drawdown kill switch constantly. Jason authorized broker-code and risk-setting changes in service of the staged goal. (2) **Fees.** `PaperBroker` now fills at quote x (1 +/- (0.2% slippage + 0.6% taker fee)), the fee being Coinbase Advanced's entry-tier (<$10k/30-day volume) taker rate on every market fill and triggered stop, folded into the fill price. Stocks (Alpaca) stay commission-free.
**Why:** The backtest (D-029) showed crypto fees alone move a strategy from marginal to losing; a paper venue without them would have produced an 8-week track record ~1.2% too good per crypto round trip, which is exactly the evidence stage 2 rests on. Folding the fee into the price keeps one code path — cost basis, realized P&L, the loss-streak breaker and cash all see it with no fee ledger.
**Consequences:** `brokers/paper.py` (`TAKER_FEE_PCT`, `FILL_COST_PCT`). Only new fills pay it — existing paper positions keep their fee-free basis. Re-check the fee against Coinbase's schedule if volume tiers ever change. Two side effects, from review: (a) the gate sizes risk on (quote - ATR stop) and ignores costs, so the loss actually realized at the stop is ~1.6 points of price larger than budgeted on crypto — at a 2x ATR stop 6-8% away that is 20-27% more than the 1% risk budget (it was already ~0.4 points before the fee); folding round-trip cost into the sizing distance is a risk/ change for Jason; (b) the tracker floors a trail's high-water mark at the fee-inclusive entry price, so any percentage trail sits ~0.8% tighter on new crypto positions (moot while no trail is live, D-032). Stop-loss and signal tuning follow as backtest studies, with results shown to Jason before live limits change.

## D-031 - Stop and signal study: the 3% trail and the ML signal are what lose money (2026-09-23)

**Decision:** Add a research grid to the backtest (`backtest/study.py`, `engine.StopRule`) and base the next live changes on its out-of-sample half. The grid: entry signal {ML model at base rate + 0.05, a 20/50-day trend filter, always-in} x ATR stop multiple {2, 3} x loss-streak pause {5, 7} x trailing rule {live 3% intraday, 3% on close, 8% intraday, 8% on close, 3x-ATR trail on close, ATR stop only}, at the paper venue's costs (0.2% slippage, +0.6% crypto fee), ranked on 2023-10-07 -> 2025-03-27 and reported on 2025-03-28 -> 2026-09-22. The ATR stop always stays intraday (it rests at the venue); only the trail is ever close-triggered. Live settings are NOT changed by this entry — the proposals below go to Jason first, as promised in D-030.
**Why / findings (past results only; corrected after the adversarial review):** The grid is 3 x 2 x 2 x 6 = 72 configurations at the 5% position cap (findings 1-8 below); a position-cap dimension {5%, 10%} was added afterwards so D-032's 10% figures are reproducible (144 in all). (1) **Robust across every signal and both halves:** the fixed 3% intraday trail is the worst or near-worst exit — model -8.1% / -4.4%, trend -9.6% / -8.1%, always-in about -14% / -15% (the always-in figures are floored by the 15% drawdown kill switch, which fired in both halves). This is the one finding that does not depend on which half is looked at. (2) **Ranked on the tune half,** the best configuration is the trend filter (close above the 50-day average, 20-day above 50-day) with a 3x-ATR trail tested on the close: +10.5% tune (Sharpe 1.39), +3.4% validate (Sharpe 0.53, max drawdown 7.0%) against buy-and-hold at equal exposure of +12.4% / +6.9%. The trend filter with the ATR stop only ranked third on tune but did better on validate (+11.8% / +5.3%, max drawdown 6.4%) — preferring it over the tune-ranked pick uses the validate half, so between the ATR-based exits the holdout does not choose. (3) **No configuration beat buy-and-hold at equal exposure by more than noise.** The only rows that edge past it are always-in with the ATR stop only (e.g. +16.9% vs +16.1%, 11 trades), which is itself a 5%-capped buy-and-hold rather than a strategy. A validate Sharpe of 0.77 over ~18 months has a standard error of about 0.8 — not distinguishable from zero. (4) **The ML model has no edge as a mechanical threshold rule** (best: about 0% tune, +1.6% to +3.1% validate, below equal-exposure buy-and-hold); whether its score helps as one input to the LLM analyst was not, and cannot be, tested here. (5) ATR 2x vs 3x differs by under a point everywhere — the study does not favour either. (6) The 5-vs-7 loss-streak pause makes no difference for the trend and model signals; for always-in the spread (14.0% vs 16.9%) is within noise. (7) Exposure is the ceiling on absolute return: the 5% position cap across six symbols holds the book to 15-35% invested. (8) A close-triggered trail exits at the NEXT open in the engine (a stock's close cannot be traded once observed); the first version of this study exited at the close itself, slightly flattering those rows.
**Proposed next, as sent to Jason:** (a) drop the fixed 3% trail — supported by (1); (b) stop showing the ML predictions to the analyst — supported only as far as (4) goes; (c) study a larger position cap. Jason decided in D-032; this entry is the evidence as it actually stands.
**Consequences:** `backtest/study.py` (CLI, report to `backtests/study-*.json`), `engine.StopRule` (default = live behaviour; the prior backtest results are unchanged), `backtest.run.model_scores`. Selection bias remains: 144 configurations were tried, and the tune/validate split limits but does not remove it. `StopRule` takes at most one trail (percent or ATR multiple); `Params.stops=None` follows the limits' own trail, so a limits change can never be silently ignored.
## D-032 - Live rules: no fixed trail, 3x ATR stop, 10% position cap; ML predictions leave the prompt (2026-09-23)

**Decision (Jason, 2026-09-23, on the D-031 study):** (1) Remove the fixed 3% trailing stop from the live rules (`RiskLimits.trailing_stop_pct = None`) and widen the ATR stop to 3x (the top of RISK_RULES' 1.5-3.0 range). (2) Stop showing the ML model's predictions to the analyst. (3) Raise the position cap to 10% of equity (the top of RISK_RULES' 5-10% range). RISK_RULES.md is updated accordingly (the trail line is struck through with this entry's reference; Layer 2 records the live 10%).
**Why:** D-031: the 3% trail was the largest loss driver for every signal tested, and the ML model showed no tradable edge once the trail was gone; exposure (capped at 15-35% of equity by a 5% per-position cap over six symbols) was the ceiling on absolute return. Backtest at the new settings (tune / validate halves, past results only): a trend filter made +23.9% / +8.8% with 7.5% / 11.1% max drawdown; always-in +41.5% / +20.1% but tripped the 15% kill switch once in the tune half; buy-and-hold at equal exposure +24.9% / +13.4% and +51.9% / +21.9%. Doubling the cap roughly doubled returns and drawdowns — it scales the bet, it does not create an edge.
**Consequences:** `risk/limits.py` (trail None, ATR 3.0, cap 0.10); `risk/stops.py` (no-trail paths: `trailing_stop_price` refuses, `effective_stop` returns the ATR stop); `portfolio/tracker.check_trailing_stops` returns [] with no trail; tests that exercise the trail mechanism now pin `trailing_stop_pct=0.03` explicitly, and new tests pin the live no-trail behaviour; `llm_analyst` prompt and whitelist test drop `predictions`. With 1% risk per trade and a 3x ATR stop, most entries are now cap-bound (a 2%-ATR stock sizes to ~16.7% at risk parity, capped to 10%). Positions opened earlier keep their 2x ATR venue stops. Drawdowns roughly double at the larger cap, so the 3% daily / 5% weekly breakers and the 15% kill switch will trip more often. Stage 1 of D-030 is still unmet by any rule the backtest can run; the next lever to study is the entry signal (e.g. whether the analyst should be steered by, or replaced with, the trend filter), which is Jason's call. From the review of this change: (a) **per-venue cash** — the 10% cap is measured on pooled equity, which exceeded the crypto paper venue's own cash ($11,000 of a $110,000 book vs $10,000 at the venue), so every crypto buy would have been refused; `PortfolioState.venue_cash` (filled by the tracker per broker asset class) now caps each buy at 99% of its own venue's remaining cash across the batch (`RiskLimits.venue_cash_usable_pct`, headroom for slippage and fees), while the 20% reserve stays pooled; (b) with 1% risk and a 3x ATR stop, most entries are cap-bound, so a stop-out costs ~0.3-0.8% of equity, but a 30% overnight gap on a 10% position costs 3% — exactly the daily halt; (c) after `/killswitch` then `/resume`, held positions have no protective stop (the kill switch cancelled them and the trail no longer covers it), fixed in D-036; (d) the study's evidence does not favour 3x over 2x ATR (they differ by under a point; for the model signal 2x did slightly better), and removing the predictions is supported only as "no edge as a mechanical rule" (D-031) — both were Jason's call on that evidence. `Settings` printing API keys in its repr is tracked separately.

## D-034 - A weekly Discord report measures the staged goal (2026-09-23)

**Decision:** The weekly report is one Discord message posted by the worker every Monday 13:00 UTC for the previous ISO week, computed from what the cycle already records: portfolio snapshots (week return, worst dip, average invested share), fills with realized P&L (closed trades, win rate, net P&L, profit factor), the risk node's decisions rows (rejections by reason, cycles that failed to assemble the portfolio) and the stored daily closes (equal-weight buy-and-hold of the watchlist over the same week). The stage-1 test of D-030 is stated explicitly: the account's return against that buy-and-hold scaled to the account's average invested share.
**Why:** Phase 3's eight weeks of paper trading are only useful if someone looks at them against the goal each week; the Discord channel is where Jason already reads the agent. Comparing with fully invested buy-and-hold would say nothing about skill for a book that is mostly cash by design (D-029/D-031), so the benchmark is exposure-matched, like the backtest's. Prediction accuracy (also asked for) is not in the report: the ML predictions no longer drive anything (D-032), and a weekly hit rate on a handful of symbols would be noise.
**Consequences:** `reports/weekly.py` (`build_report`, `format_report`, CLI), the worker's `weekly_report` task; the risk node now records `trading_mode` in its decisions detail so the report can scope cycles by mode. Shaped by the adversarial review, which simulated the first real Monday and found it would have announced a +1000% week (the crypto venue joining the book) and a 'BEAT it' verdict while nothing was invested: (a) a step in equity larger than the invested positions could move is treated as a cash flow and the week's return is reported as not measurable; (b) the comparison is withheld below 5% average exposure, is stated as points ahead/behind rather than a verdict, names a losing week, and says one week is noise; (c) the invested share is time-weighted (the agent's snapshots are sparse when it is not running); (d) a watchlist symbol without closes on both sides of the week is left out of the benchmark and named, instead of counting as 0%; (e) the prior-week anchor is used only if it is from the prior week; (f) a failed build posts `weekly report FAILED: <type>` to Discord. Snapshots of zero equity (before any venue was configured) are ignored.
## D-035 - Crypto daily candles are bucketed by bar open; crypto re-fetches its whole window (2026-09-23)

**Decision:** `CoinGeckoClient.get_daily_candles` buckets each /ohlc bar by its OPEN time (stamp minus the bar length: 30 minutes for 1-2-day windows, 4 hours for 3-30), drops the window's first day when it starts mid-day, and assigns market_chart's trailing-24h volume stamped at D+1 00:00 to day D. Crypto ingest fetches its full 30-day window every cycle instead of resuming from the latest stored bar.
**Why:** the crypto skew D-024 recorded, reproduced live 2026-09-23: over 12 BTC 4h bars, CoinGecko's close matched the market at the bar's stamp to 0.05% on average while its open matched the market four hours earlier — the stamp is the bar's CLOSE. Bucketing by the stamp had made every stored crypto "day" run 20:00 -> 20:00 UTC, putting stored closes ~0.4% (BTC) to ~1% (DOGE) away from true daily closes; the risk gate's crypto ATR, the liquidity filter and every crypto feature read those rows. Separately, the incremental cursor asked for a 1-day window (30-minute bars covering the last 24 hours), whose first UTC day was partial and was upserted over yesterday's complete bar every cycle. Fetching the full window costs the same two requests per symbol, heals the last 30 days on the next cycle, and matches how stocks already work (D-021). Jason authorized data-layer changes.
**Consequences:** after the fix, stored daily closes match Alpaca's 00:00 UTC closes to 0.01% on average (live check, 29 days). Rows older than the 30-day window keep the old buckets, the old one-day-late volume mapping, and whatever partial data the old cursor wrote over them; ATR (15 bars) and liquidity (20) heal on the first cycle, the 60-bar features within about a month (deleting those rows is irreversible, so it was not done). A day whose volume point is not published yet (yesterday, just after midnight) is left out rather than written with volume 0. `ingest._fetch_start` and `OhlcvRepository.latest_ts` are gone (the crypto cursor was their only production use). Live review 2026-09-23: 30-minute bars are close-stamped too, the newest bar is always a finished one, 4h bars align on 00/04/08 UTC, and the output matched Alpaca's daily closes and opens to 1.1bp on average.
## D-036 - The executor re-protects any held quantity that no stop covers (2026-09-23)

(Drafted as D-033; renumbered at merge so the log stays in merge order - there is no D-033.)

**Decision:** Each cycle, after reconciliation and before any submission, the executor sums the open sell-stop rows in its own orders book per symbol and compares them with each venue's held quantity; any uncovered remainder (a whole position after /killswitch then /resume, or an added lot whose stop failed) gets a standalone GTC stop at this cycle's price - N x ATR (the gate's own formula and limits) through a new `Broker.place_stop` (PaperBroker, Alpaca whole-share stop, Coinbase stop-limit) and is booked like an entry's protective stop. It runs when not halted and under the halts that leave protective stops working - breaker cooldowns and the automatic drawdown kill switch in the cycle that trips it (later cycles are skipped outright; D-019, D-020) - which amends D-016's "nothing is submitted under any halt" for protective stops only. It never runs under the manual kill switch (cancel means cancel), with Redis unreachable (a human may have armed the switch; fail closed), at a venue whose reconciliation failed (a stale book could hide an existing stop), or for a symbol this cycle closes (trail breach or approved sell - unless halted, when nothing is submitted and so nothing closes). With no price for the symbol this cycle (stocks out of session, a failed quote, a holding outside the watchlist) placement is deferred with a log line only: a stop sized from a stale close can sit above the market after a gap and sell at once. A stop that cannot be computed or placed logs `position_unprotected` at error level (pages) and retries next cycle; a stop placed but not booked is reported as `placed_unbooked` (like `_book`), never crashing the executor.
**Why:** `/killswitch` cancels every open order including the protective stops (D-020) and `/resume` never re-placed them; until D-032 the tracker's 3% trail would still have closed such a position, but with no fixed trail it could stay open with no exit rule at all. Checking coverage by quantity against our own book each cycle - rather than re-placing on /resume only - also covers an added lot whose stop failed. The adversarial review drove the quantity check, the fail-closed Redis rule, the broad per-position containment, the no-price deferral and the approved-sell exclusion. Jason authorized executor/broker changes.
**Consequences:** `Broker.place_stop` on all three brokers; `trade_executor._may_reprotect/_reprotect/_protect`; Discord label `(re-protect stop)`. Known limits: a stop that exists at the venue but not in our book (an OTO leg id missing from the response, a stop replaced by hand) gets a duplicate the venue refuses for lack of free shares - paging each cycle until a human looks; a holding outside the watchlist has no price, so it is never protected and only logged (`reprotect_deferred`) - the agent never trades such symbols, so they are the human's own; a watched symbol with no stored candles cannot get an ATR stop and pages each cycle; Coinbase does not look a timed-out stop up by its key (Alpaca does), so a stop accepted behind a timeout is reported unprotected and its duplicate pages; the book records the requested quantity and stop, which the venue may round (Alpaca whole shares and ticks up, Coinbase increments).

## D-037 - The in-flight cycle id lives in Redis, so a process death is resumed on restart (2026-09-23)

**Decision:** The scheduler records the cycle in flight under `agent:cycle_in_flight` in the kill switch's Redis (TTL 2h): the id while its first run is going or after it crashed, `<id>:resumed` during its one replay. The key is cleared when a cycle ends (or is abandoned). A run that finds a plain id replays that cycle; a run that finds `<id>:resumed` abandons it with an error-level `cycle_abandoned` (pages) and starts fresh. If Redis fails on the marker alone, the cycle still runs under a fresh id (warning `cycle_marker_unavailable`); it just cannot be resumed - and a marker Redis failed to overwrite or delete stays until its TTL, so the next run may replay (or re-abandon, paging again) that older cycle; a finished cycle replays as a no-op.
**Why:** It closes D-019's known gap. The resume-once id lived in process memory, so a process that died between a venue submission and its booking (a power cut, or a closed terminal) came back under a fresh id, and the fill surfaced as an `untracked` position. Now the restart replays the unfinished nodes under the same idempotency keys, and the venue returns the existing order instead of a second one. The `:resumed` suffix stops a cycle that kills the process on every attempt from being replayed forever. The 2h TTL outlives the 1h crash pause, but it does not let a cycle whose approved orders were sized on prices many hours old be replayed.
**Replays only where they matter (review):** a restart can now come up to 2h after the crash, and replaying fetch -> analyst -> gate on a checkpoint that old would size orders and set stops from stale prices (a stop above the market sells at once, the same hazard D-036 defers re-protect for). So `run_cycle` resumes a crashed cycle only when its next node is the executor or later - the only case with submissions to book. A cycle that crashed earlier submitted nothing and is REPLACED by a fresh cycle under a new id (warning `cycle_replaced_after_crash`). This also changes the in-process crash path (it used to replay the analyst an hour later on the old prices). The executor replay still resubmits the cycle's not-yet-sent approved orders as sized at the crash, up to the TTL later - accepted: it is bounded, rare, and every order still carries its stop.
**Consequences:** `scheduler.AgentRunner._claim_cycle/_release_cycle`, `KillSwitch.redis`, `builder.RESUME_FROM`. Behaviour within one process is unchanged: a crash is resumed once, and a second crash abandons it. As before, a cycle held back by a kill switch keeps its marker, and it is replayed if the switch is cleared within the TTL. Two agent processes on the same Redis would share the marker. Only one ever runs, and `--once` does not use the runner.

## D-038 - Claude's entries are steered by the 20/50-day trend filter; the gate never adds to a held position (2026-09-23)

**Decision (Jason, 2026-09-23: "steer Claude with the trend filter"):**
(1) The feature engine adds `trend_up` to each symbol's features: 1 when, at the last COMPLETE daily close, the close is above its 50-day average and the 20-day average is above the 50-day (the study's rule, D-031); 0 otherwise; absent under 50 complete bars. Complete means dated before the cycle's UTC day, so the label is fixed for the day, like the backtest's decision on a finished close. A test pins the label equal to `study.trend_scores` on every day.
(2) The prompt tells Claude: buy only when trend_up is 1, lean toward buying then unless the other data argues against it (and say what did), propose sell when it is 0, and do not sell when it is 1.
(3) `ClaudeAnalyst` DISCARDS any buy whose trend_up is not 1 and logs the discarded proposal with its confidence and reasoning (`analyst_buy_against_trend_dropped`). The prompt is guidance and the filter the guarantee.
(4) **Risk gate: a buy for a symbol already held is rejected `already_held`.** The tested strategy buys only when flat (`backtest.engine._proposals`).
**Why:** The backtest ranked this filter best among the rules it can run, but none beat equal-exposure buy-and-hold by more than noise (D-031, D-032), and Claude is the one part it cannot test. Steering keeps Claude's judgment as a veto on entries on top of the rule. The review of the first version found two problems:
- Labelling from today's still-forming bar would flip the label intraday around the 50-day average (sell at 10:00, buy back at 10:15), a round-trip cost the backtest never pays. Hence (1)'s complete bars.
- "Lean toward buy" every 15 minutes on a held stock made the gate size a sub-share top-up that Alpaca rounds to 0 and the executor pages as a permanent rejection, every cycle, for every held stock. On a dip it would have averaged down, which was never tested. Hence (4).
(4) makes the gate stricter; it never loosens a limit. The 10% cap, 1% risk, reserve and breakers are unchanged.
**Consequences:**
- Code: `features.engine.trend_up`, the feature node, the prompt rules, the analyst filter, and `RejectReason.ALREADY_HELD` (in the weekly report's wording).
- The gate's existing-position headroom arithmetic now only ever sees 0 for held value; it is kept as-is rather than rewritten in this change.
- **Crypto gets no buys until it has 50 complete daily bars.** On 2026-09-23 BTC/ETH had 39, so buys start around 2026-10-05. For the next ~5 weeks about 9 of those 50 bars are the older 20:00-20:00 UTC rows D-035 left in place (closes 0.4-1% off true daily closes), so early crypto labels can differ slightly from what the study measured.
- Stocks all read "up" on 2026-09-23: expect one buy per symbol (each up to the 10% cap), then `already_held` rejections while held. Downtrend sells of unheld symbols add `no_position_to_sell` rejections. Both are counted in the weekly report and never posted per cycle.
- **Before crypto goes live:** Coinbase reports any positive balance as a position, so leftover dust (rounding, a partial fill, coins bought by hand) would make every later buy of that coin `already_held`, and a trend-down sell of dust can fall below the exchange minimum. The paper venue leaves no dust. Needs a minimum-size threshold at that point.
- Whether Claude's veto adds anything over the plain rule is what the 8-week paper period will show; past backtest results do not guarantee it.

## D-039 - Core + swing study: the account-wide 15% kill switch cannot sit over a 70%-invested core (2026-09-24)

**Decision:** Add `backtest/core_swing.py` (`python -m trading_agent.backtest.core_swing`, ~8s), the first step of the core + swing work, which models the plan Jason chose on 2026-09-24:
- **Deposits and split:** a starting deposit, then a fixed deposit every 14 days. 70% of each deposit goes to the CORE: VOO 37.5% / QQQM 22.5% / BTC 6% / ETH 4% of the account, filled at the next open with fractional units. Each holding's shortfall below target is filled in proportion, and crypto is never bought above 10% of the account. 30% goes to the SWING sleeve: 13 stocks trend-traded through the LIVE `risk.gate.evaluate`, with whole-share fills.
- **Grid:** 5 core-stop rules x 2 kill-switch scopes, ranked on a tune half (2021-10-01 -> 2024-03-27) and reported on a validate half (2024-03-28 -> 2026-09-23).
  - Core-stop rules: none, 3x ATR, 15% trail, 25% trail (all checked on closes), and sell below the 200-day average.
  - Kill-switch scopes: stops everything, or stops the swing sleeve only.
- **Benchmark:** every deposit into VOO.

**Nothing live changes.** This entry is the evidence for Jason's sign-off on the plan's items 1-6.

**Semantics fixed here, which step 2's live accounting must match:**
- (a) **Anchors.** A deposit scales every equity anchor the gate reads (peak, day start, week start) by its share of the account. A deposit is then neither a gain nor a dilution of a loss already taken, and the drawdown kill switch sees exactly the time-weighted drawdown the report prints.
- (b) **Swing sizing.** The gate sees whole-account equity, so risk per trade is 1% of the account. The 25% position cap and 20% reserve are rescaled to the swing sleeve's own equity.
- (c) **Core re-entry.** A stopped-out core holding re-enters only on a CROSS back above its 200-day average: a close at or below it, then one above. Otherwise a stop that fires above the line buys back two days later, and every stop becomes a round trip.

**Findings (past results only; paper costs; money-weighted yearly return, MWR / time-weighted max drawdown):**
1. **The 15% account-wide kill switch trips in ordinary corrections.** It tripped in the tune half for every core-stop rule (February-April 2022). In the validate half it tripped for every rule except 3x ATR and below-200-day (April 2025).
   - With scope "everything", deposits then pile up in cash. Tune MWR fell to between -1.4% and +3.8%, against +14.1% to +21.2% with scope "swing only".
   - The live kill switch is durable until a human `/resume`s, which the backtest cannot model. It halts the swing sleeve for the rest of the half.
   - A 70%-invested buy-and-hold core and an account-wide 15% kill switch do not fit together. Candidates for Jason: scope the kill switch (and the breakers) to the swing sleeve's own equity, and never stop core buying.
2. **Core stop, scope "swing only":**

   | Core stop | Tune MWR / maxDD | Validate MWR / maxDD |
   |---|---|---|
   | No stop | 21.2% / 27.1% | 13.8% / 19.9% |
   | 3x ATR | 20.1% / 21.5% | 9.2% / 11.8% |
   | 25% trail | 18.8% / 22.4% | 13.8% / 18.2% |
   | Below 200-day | 17.5% / 17.6% | 10.7% / 12.5% |
   | 15% trail | 14.1% / 18.3% | 10.3% / 16.8% |
   | **VOO, every deposit** | **16.5% / 25.1%** | **17.8% / 19.1%** |

   - Wider exits keep more return: the 25% trail matched no-stop on validate with a smaller drawdown.
   - The 200-day rule and 3x ATR cut drawdowns the most, and cost 3-5 points a year on validate.
   - The plan beat VOO on the tune half (QQQM and crypto's 2023 recovery) and lost to it on validate under every rule.
3. **The swing sleeve lost to holding in both halves.** Swing MWR was 0.8% to 4.1% on tune and -1.9% to -0.1% on validate, against VOO's 16.5% / 17.8%. Its run was not a fair one:
   - it was halted by the kill switch for most of each half;
   - at a $300-$10,000 sleeve, approved buys failed to afford one whole share on 316-485 days;
   - the reserve refused 559-884 more.

   Swing trading at this account size has no demonstrated edge.

**Consequences:**
- New code: `backtest/core_swing.py`. `backtest/metrics.py` gains `money_weighted_return` (XIRR), `time_weighted_drawdown` and `deposit_and_hold`. Reports go to `backtests/core-swing-*.json`.
- The module imports engine's private helpers (`_Holding`, `_portfolio`, `_proposals`, `_window`), accepted as same-package coupling.
- Not simulated: Claude's veto on swing entries, Layer 5 (VIX/macro), a human `/resume`, and taxes.
- History starts at Alpaca's IEX coverage (2020-07-27; QQQM 2020-12-02; crypto 2021-01-01), so the study begins once every 200-day line exists (2021-10-01).

## D-040 - Jason's core + swing rules; the paper ledger simulates the deposits at real scale (2026-09-24)

**Decision (Jason, 2026-09-24, on D-039: "go with your recommendations, keep the swing portion"):** the plan's design points:
1. **Core stop:** a 25% trail on the close. Re-entry on a cross back above the 200-day average.
2. **Position cap:** the core follows its target weights. The 10% position cap is not applied to it.
3. **Crypto cap:** deposits stop buying crypto above 10% of the account. Nothing is sold to trim it.
4. **Cash reserve:** 20%, inside the swing sleeve only.
5. **Swing sizing:** risk per trade is 1% of the whole account; each position is capped at 25% of swing-sleeve equity; whole shares only.
6. **Breakers and kill switches:** the daily/weekly breakers, the loss-streak pause and the drawdown kill switch watch the swing sleeve only, and core buying never stops for them. The manual `/killswitch` still stops everything.
7. **Swing sleeve:** kept.

These take effect with steps 3-4 (D-041, D-042), and RISK_RULES.md changes with them. Nothing in the live gate changes here.

**Step 2 (this entry's code): the paper ledger.** Paper venues come with play money (Alpaca paper ~$100k; the crypto PaperBroker $10k), and Alpaca paper accounts cannot receive transfers.
- **Starting it:** `python -m trading_agent.portfolio.ledger start` (paper only; refused while any venue holds positions) writes a `cash_flows` "baseline" row per venue that cancels the venue's own cash. It then books the first deposit, split 10% crypto / 90% stock: the crypto cap's share of every deposit lands at the crypto venue.
- **What the app owns:** from then on, what the app owns at a venue = the venue's figures + its ledger rows (`PortfolioTracker.build_state`).
- **Later deposits:** the tracker books each deposit that has come due (every 14 days from the start). A unique index on (mode, venue, kind, date) means a date can never be booked twice.
- **Anchors (D-039 (a)):** snapshot history before the start is ignored. Every anchor is scaled by (equity_before + amount) / equity_before for each deposit booked after its snapshot, so a deposit is neither a gain nor a dilution of a loss.
- **Peak:** the peak is now the LATEST snapshot's stored peak times that growth. Without deposits this equals the old max(), because stored peaks only rise.
- **With no ledger rows,** every path behaves exactly as before.
- **Weekly report:** it takes ledger deposits out of the week's return and drawdown, and prints "Deposits this week" and "Since <start>: $X deposited, now worth $Y".
  - Deviation from the plan text: it shows the dollar gain on deposits rather than an annualized money-weighted return. Annualizing a few weeks of deposits produces meaningless figures. The money-weighted return stays in the backtest (D-039).

**Why:** Without this, a deposit would read as a gain to the breakers and the peak, and a withdrawal as a loss that trips the daily breaker. And the paper period would run on $110k of play money, never meeting the small-account limits (whole shares, the $300 swing sleeve) the study found decisive.

**Consequences:**
- New code: `portfolio/ledger.py` (the plan's deposit constants, now also imported by the backtest), `CashFlow` / `CashFlowRepository`, and `SnapshotRepository.snapshot_at` / `latest`, which replace `equity_at` / `peak_equity`.
- **The ledger is NOT started by this change.** Starting it needs:
  - the paper positions closed (Alpaca paper held AAPL, MSFT, NVDA and SPY on 2026-09-24);
  - steps 3-4, because at the first deposit's size today's rules (10% cap, whole shares) would make nearly every buy unaffordable.
- **Live (Phase 4)** needs deposit detection at the venue instead. A live deposit is real money at the venue, so it must scale the anchors without being added again.
- **From the review:**
  - Only the risk node books deposits (`build_state(book_deposits=True)`, snapshotted at the same `now`). A `/status` call that booked one mid-cycle would have put the deposit inside an earlier-stamped snapshot. The peak would then count it twice and trip the kill switch.
  - Configured venues must match the ledger's baselines, or the cycle fails closed. A venue added later would otherwise count its play money as the app's.
  - The loss streak also ignores trades from before the start.
  - **Operating rule: stop the agent and cancel open orders before `ledger start`.** A cycle snapshotting the old $110k account after the start would become the new peak.
  - A restart can't happen on a day the old ledger booked a deposit; the CLI says so.
  - A weekly report for a week before a restart reads that week as another account.

## D-041 - The core sleeve: deterministic deposit buys and a close-based 25% trail; a position's sleeve is its symbol (2026-09-24)

**Decision:** Step 3 of the plan implements D-040 items 1-3 in `portfolio/core.py`, planned by the risk node whenever a paper ledger is running:
- **Buys:** 70% of deposits buy the core toward `CORE_TARGETS` (VOO 37.5%, QQQM 22.5%, BTC 6%, ETH 4% of the account). This is the single source; the backtest imports it too.
  - Each holding's shortfall is filled in proportion, and a holding that cannot be bought now keeps its share in cash. Buys under $10 wait (see below).
  - Each venue pays only from its own cash. The crypto venue's app cash is all core, capped so that crypto never exceeds 10% of the account; nothing is sold to trim it. The stock venue's core cash is 60% of everything deposited, minus core stock buys (fees included), plus core stock sales, minus open core buys.
  - Stock buys are fractional DAY market orders at Alpaca (Alpaca takes fractions on DAY orders only). They happen only while the NYSE session is open, because stocks have a price only then.
- **Stop:** a holding sells in full when the last COMPLETE daily close is at or below 75% of the highest complete close since the position opened. That high is floored at the average cost, and at the cost alone when the book has no fill history for the position. There is no venue order: Alpaca stops take whole shares only. Re-protect skips core symbols while a ledger runs. A core sell goes out even when its venue's reconcile failed, like a breach close: a full close never depends on the book.
- **Re-entry:** a sold holding is bought again only after a complete close at or below its 200-day average, then a later one above it (D-039 (c)). A sale before the ledger started never blocks buying.
- **The analyst never sees or trades a core symbol.** Such a proposal is dropped before the gate (warning `analyst_core_proposals_dropped`). The core's orders join `approved_orders` with no stop and no source proposal, and are audited under `audit["core"]`.
- **Open orders:** an open core sell counts as a sale already made. The executor books a sell's fill only on a later cycle, so until then the book still shows the buy while the venue shows no position. Found in review: without this the next cycle bought the holding straight back, skipping the 200-day re-entry. A holding with an open sell is not sold twice.
- **Halts:** no core orders are planned under any halt, because the executor refuses them and would post a HALTED line every cycle. Step 4 lets them through cooldowns and the automatic drawdown kill switch.
- **Re-protect** skips core symbols only while a ledger runs. Without one, a core symbol is an ordinary position and keeps its venue stop.
- **Small buys:** buys under $10 wait. The 1% left by `venue_cash_usable_pct` would otherwise become a second round of dust orders after each deposit.
- **Containment:** a failure planning the core logs `core_plan_failed` (it pages) and places no core orders that cycle. It never stops the swing sleeve's gate.
- **CoinGecko quotes:** one `/simple/price` call now prices every mapped coin and serves quotes for 30 seconds. In the end-to-end run, the free tier 429'd the ETH quote and the paper venue's fill quote, which comes seconds after the cycle's own. Crypto core buys failed until quotes were shared.
- **Watchlist:** stocks are VOO, QQQM + the 13 swing names. SPY is dropped. Crypto is BTC-USD, ETH-USD, core only.

**Deviation from the plan text ("an internal book splits each symbol into core and swing quantity"):** a position's sleeve is decided by its SYMBOL. The core list and the swing list never overlap: funds and crypto in one, single stocks in the other. So venue positions, fills and orders need no split, and the acceptance rules ("swing never sells core quantity", "a core stop never sells swing quantity") hold by construction. A quantity book would be a second source of truth, needed only if a symbol were ever in both sleeves, which Jason's lineup rules out. If that changes, a split book is the prerequisite.

**Known gaps:**
- **Crypto re-entry:** BTC/ETH have fewer than 200 stored daily bars (the CoinGecko 30-day window, D-035). A crypto stop-out therefore cannot re-enter until about April 2027 unless the history is backfilled, e.g. from Alpaca crypto bars, which match 00:00 UTC closes. Follow-up: D-046.
- **Dividends:** VOO/QQQM dividends land in the stock venue's cash but count as swing cash, because core cash is computed from deposits and fills.
- **Crypto over the cap:** crypto cash cannot move to stocks when crypto is over the cap, since the venues are separate. The backtest pooled it.
- **Order type:** core stock orders are market orders, not limits near the quote as the plan suggested, and only during the NYSE session. A limit would need handling for partial fills and expiry for little benefit at this size.
- **Swing can spend core cash until step 4:** the gate still sizes swing entries against all of a venue's cash. The ledger must not be started before step 4 (the swing sleeve's own equity and cash) is merged.
- **Halts:** the executor still refuses every order under any halt, core included. Step 4 lets core orders through cooldowns and the automatic drawdown kill switch (D-040 item 6).
- **Unbooked sells:** if a core sell's order row cannot be written (`submission_unbooked`, which pages), its fill is never booked. The next cycle then sees the symbol as never sold and buys it back, bypassing the 200-day re-entry. A human fixes the book.
- **Symbols held back from the analyst:** VOO, QQQM, BTC and ETH are hidden from the analyst from the moment this merges, ledger or not. Until the ledger starts nothing trades crypto; any crypto held keeps its venue stop.
- **SPY:** the new watchlist drops it, so SPY gets no price and the agent no longer manages it. The paper account holds 14 SPY with its GTC venue stop at Alpaca, but the analyst can't sell it and re-protect defers it. It must be closed at the cutover (with the other paper positions) or by hand.
- **The backtest keeps its own allocation** (`core_swing.core_allocation`): it pools crypto cash into stocks. Only the targets are shared.
- **Test data:** integration tests ingest `itest` candles for the fixed core symbols (VOO) into the shared database for the length of the run. Teardown deletes them; a live cycle running at that moment could read them.

## D-042 - The swing sleeve in the risk gate: its own equity, reserve, cap and breakers; the core trades through swing-only halts (2026-09-25)

**Decision:** Step 4 of the plan implements D-040 items 4-6. With a paper ledger running, the gate evaluates the SWING SLEEVE's own portfolio (`portfolio/swing.py`), built by the risk node each cycle:
- **Swing equity** is the stock venue's app cash minus the core's stock cash (`core.stock_core_cash`), plus every non-core position.
- **Sizing:**
  - `SWING_LIMITS` caps a position at 25% of swing equity.
  - The 20% reserve is measured on swing equity.
  - Risk per trade stays 1% of the WHOLE account, via the new `PortfolioState.risk_equity`, which sizing uses when set.
- **Whole shares:** the gate now rounds stock buys down to whole shares. A buy under one share is refused `position_size_zero` ("under one whole share"). Before, Alpaca rounded it to 0 and the executor paged a permanent rejection every cycle. This applies with or without a ledger. The backtest counts those as gate rejections now instead of "unaffordable", and batch cash accounting is tighter by the rounding.
- **Breakers measure the sleeve's own returns:**
  - Each snapshot stores `swing_equity` and a `swing_index`, a time-weighted value that moves with the sleeve's gains and losses but not with its share of deposits (the stock rows less the core's 60%, exactly as `stock_core_cash` splits them).
  - An anchor (day start, week start, peak) is swing equity x index_then / index_now. So a deposit neither hides nor dilutes a loss already taken, the same property as D-039 (a) for the account.
  - The loss streak counts swing closes only (`trailing_loss_streak(exclude=core symbols)`).
- **Halts** (`risk.killswitch.protective_orders_allowed`):
  - Breaker cooldowns and the automatic drawdown kill switch stop swing entries only. The core keeps buying, and its trail keeps selling: the executor submits core orders under them, and the risk node plans them.
  - The manual `/killswitch` and an unreachable Redis stop everything.
  - The scheduler no longer skips cycles under `kill_switch:max_drawdown_exceeded`.
  - The analyst is not consulted under any kill switch (no LLM spend while nothing could pass).
  - The gate arms the drawdown switch only when no kill switch is set. Re-arming every running cycle would page each time, and overwriting the human's switch would downgrade "stop everything".
- **Without a ledger nothing changes** except whole-share rounding, the analyst skip, and cycles running under the automatic drawdown switch (they self-reject; stops keep working).

**Why:** Jason's rules (D-040), and D-039's finding that an account-wide 15% kill switch over a 70%-invested core trips in ordinary corrections and then leaves every deposit in cash.

**Consequences:**
- `portfolio/swing.py`; `snapshot.swing_equity` / `swing_index` (added by `init_db`); `SnapshotRepository.peak_swing_index`; `PortfolioState.risk_equity`; `risk.killswitch.protective_orders_allowed` (was the executor's `_may_reprotect`).
- If the core's inputs cannot be read, the risk node fails closed (no orders), because the sleeve cannot be measured.
- **From review, before merge:**
  - **Marks, not cost:** swing positions are valued at this cycle's price, else their last complete close. Stocks have no price outside the session, so valuing at cost would have dropped the sleeve to cost every evening, which reads as a loss or, against an intraday peak, a drawdown.
  - **Reconcile first:** with a ledger, the risk node reconciles open orders before it measures. Last cycle's core fills are then on the book, so an unbooked core sell doesn't read as a swing gain (and a fake peak, which would make the kill switch permanent) and an unbooked buy doesn't read as a swing loss.
  - **Open core buys:** they stay in the core's claim, because their cash is still at the venue.
  - **Stale venue:** if a venue's book cannot be reconciled, the sleeve is not measured that cycle. The last measurement stands, and the snapshot records no swing figures (`SnapshotRepository.latest_swing` skips it).
  - **Floor on the index:** the index never goes below 1e-9. A wiped-out sleeve reads as a total loss, and later returns still move it; at 0 the breakers would have been off for good.
  - **Crypto cash:** the view gives the swing sleeve no crypto-venue cash (it is all core).
  - **Analyst:** it is skipped under any halt.
  - **Constant:** the automatic kill switch's reason is one constant, `AUTO_KILL_SWITCH_REASON`.
  - **Second review:**
    - The day and week anchors come only from measured snapshots (`snapshot_at(swing_only=True)`). An unmeasured cycle just before midnight would otherwise have blinded the daily breaker for the whole next day.
    - A held swing symbol with no live price is marked at its newest bar only when this cycle's ingest refreshed it. Otherwise the sleeve goes unmeasured, since an old close fakes a move and then undoes it.
    - "Stale" now means the STOCK venue failed to reconcile. An unmeasured sleeve gets no cash, so it can't buy on numbers that may hold the core's money.
    - Swing cash is no longer clamped at 0. Clamping turned part of a deposit into a loss.
    - A sleeve that was never funded reads as no loss; only a wiped-out one (index at the floor) reads as a total loss.
    - The fills this reconcile books, and the stale reasons, are in the audit (`audit.swing`).
- **Known limit, small accounts:** at about $300 of swing money, a 25% cap is about $75, so a stock above about $75 can't be bought in whole shares until the sleeve grows. Expect `position_size_zero` rejections; they are counted in the weekly report, never paged.
- **Known limit, carried over from the account-level switch:** the peak never resets. After a `/resume` the next cycle re-arms the drawdown switch until the sleeve is back within 15% of its peak, so `/resume` alone doesn't restart swing trading after a 15% fall. The open question (a cooling-off expiry, or resetting the peak on `/resume`) is Jason's.
- **Also a known limit:** swing SELLS stay refused under a cooldown or the automatic kill switch (D-016), as before. Swing exits during a halt rely on the venue ATR stops.

## D-045 - Path B: no swing trading; the swing share joins the core's VOO under its 25% trail (Jason, 2026-09-26)

**Decision (Jason, 2026-09-26, on the "Swing sleeve option (b)" handoff doc: Path B; then "join the core, 25% trail"):**
- The swing sleeve stops trading. The 30% of each deposit that went to it now buys VOO as part of the core.
- `CORE_TARGETS`: VOO 67.5% (37.5% + 30%), QQQM 22.5%, BTC 6%, ETH 4%. Every deposit is core (`CORE_SHARE` 100%, `CORE_STOCK_SHARE` 90%). The crypto cap stays at 10%.
- That VOO follows the core's rules: the 25% trail on the close, and re-entry on the 200-day cross.
- **The stock watchlist is the core only (VOO, QQQM).** The analyst sees no swing symbols, so no Claude calls are made.

**Why:** the backtest evidence:
- A backtest where swing never trades matched the best result in both halves.
- Option (a)'s validate win came from its kill switch parking the sleeve, not from trading.
- Under option (b), swing trading cost return (validate 16.5% vs 19.4%).
- No backtest shows the swing sleeve adding return.

**Backtest of the chosen build** (2026-09-26, `python -m trading_agent.backtest.core_swing` with the new targets; tune 2021-10-01 to 2024-03-28, validate 2024-03-29 to 2026-09-25; MWR / time-weighted max drawdown; past results only):

| Config | Tune MWR / maxDD | Validate MWR / maxDD |
|---|---|---|
| **Path B as built: swing share in core VOO, 25% trail** | 21.2% / 29.8% | 19.8% / 20.6% |
| Swing share in VOO with no stop (the handoff's backtested Path B; one-off run on the same window) | 22.7% / 27.2% | 19.8% / 20.6% |
| Every deposit into VOO | 16.5% / 25.1% | 18.1% / 19.0% |

- The trail costs 1.5 points a year and 2.6 points of drawdown in the tune half. It sold VOO in the 2022 fall and bought back only after the 200-day cross. The validate half is identical.
- Both beat every-deposit-into-VOO in both halves. The drawdowns are deeper because of QQQM and crypto.
- The handoff asked for these numbers before merge; they are here and on the PR.

**What stays:** the swing machinery (the sleeve's view, index and breakers in the gate; the analyst; the trend filter).
- With no swing money it measures an empty sleeve, which reads as no loss (D-042). The gate sees no proposals.
- Only the watchlist and the targets changed. Adding a stock to `STOCK_WATCHLIST` would hand it to the analyst again, but with no swing cash its buys are refused.
- Removing the sleeve would be a separate, larger change. The handoff asked to remove only dead code, and none is dead.

**Consequences:**
- The breakers and the drawdown kill switch watch the (empty) swing sleeve, so nothing trips them. The core has only its trail, and the manual `/killswitch` still stops everything.
- The backtest's swing-mechanics tests run on D-040's split through a fixture (`swing_funded`), as do the database-backed swing-sleeve tests.
- `core_swing`'s report header shows the split from `CORE_SHARE`. Its swing return is 0 when no money goes to swing (it printed -99%).
- **The earlier cash-sweep build (parking idle swing cash in IVV) is superseded:** with no swing cash there is nothing to park. Its backtest code stays as the evidence.
- RISK_RULES.md is updated in this PR on Jason's instruction (2026-09-25: "apply the changes for me").
- **Added after review (2026-10-01):**
  - The core-only watchlist also narrows everything else that reads `STOCK_WATCHLIST`. The worker's weekly retrain now trains on VOO, QQQM, BTC and ETH only; `backtest/run.py` and `backtest/study.py` run on those four, so they no longer reproduce D-031's recorded numbers; news scoring and the weekly report's buy-and-hold benchmark use the core basket.
  - The worker has to be rebuilt after the change (`docker compose up -d --build worker`): it runs the code baked into its image.
  - The swing sleeve is empty only because the stock venue's 90% share of each deposit equals `CORE_STOCK_SHARE`. With a single venue (no crypto venue configured), the stock venue gets the whole deposit and 10% of it would read as swing cash. Paper always configures both venues; revisit before a live setup without crypto.

## D-046 - Crypto bars and quotes from Alpaca; `ledger start --close-positions` is the paper cutover (2026-09-29)

**Decision (Jason, 2026-09-29: "place 1 through 4 in one PR" - the cutover, crypto quotes, crypto daily bars, the macro calendar):**
1. **Crypto quotes from Alpaca.** Whenever the Alpaca keys are configured, the cycle's crypto prices and the paper venue's fill prices come from Alpaca's latest trade (`AlpacaDataClient.get_quote`). CoinGecko's free tier refused 22 of 60 BTC/ETH quotes over the 30 cycles before 2026-09-30 06:15 UTC. CoinGecko remains the keyless fallback.
2. **Crypto daily bars from Alpaca.** Ingest fetches a year of daily crypto bars (365 on 2026-09-30) instead of CoinGecko's 30 days, re-read every cycle like stocks. The core's 200-day re-entry line now exists for BTC and ETH, so a crypto stop-out can re-enter. The trend label also no longer waits for 50 stored bars.
   - Alpaca's 00:00 UTC closes match the buckets D-035 made CoinGecko's.
3. **Crypto volume is USD from every source.** Alpaca's venue volume (base units) is converted at the provider (x close), as the liquidity filter expects.
   - Alpaca's single venue trades far less than the market-wide aggregate the $50M crypto floor was calibrated on (BTC about $59k/day on 2026-09-29). A crypto buy through the risk gate would be refused as illiquid.
   - Since D-045 no crypto order goes through the gate (core buys skip it), so the floor is unchanged. Revisit it if swing trading ever returns.
4. **The cutover as one command:** `python -m trading_agent.portfolio.ledger start --close-positions`, with the agent stopped, during the NYSE session.
   - It cancels every open order, sells every position at every venue, waits up to 5 minutes until all are flat, then starts the ledger.
   - The sells are not booked, since the ledger starts flat and ignores everything before its start. The cancelled stop rows retire at the next reconcile.
5. **The macro calendar is not extended (blocked).** On 2026-09-29 the BLS had published CPI and Employment Situation dates only through December 2026 (its 2027 schedule page returned 404). FOMC dates already run to 2027-12-08.
   - `test_shipped_calendar_is_not_about_to_run_out` goes red around 2026-11-04. Extend the calendar from the BLS 2027 schedule once it is out; dates are never guessed.
   - Since D-045 the macro pause gates nothing that trades, so a stale calendar costs only the warning.

**Why:** Items 1-2 give the no-swing plan reliable crypto prices and the 200-day history its re-entry rule needs. Item 4 turns the four-step cutover into one step that can't be half-done: `start` still refuses while anything is held.

**Consequences:**
- `AlpacaDataClient.get_quote`; crypto `Candle.volume` from Alpaca is USD; ingest's `_fetch_crypto` takes both clients; `make_deps` prices the paper venue from Alpaca when configured.
- `ledger.close_positions`; `CLOSE_TIMEOUT` 5 minutes, polled every 5 seconds.
- **Fixed in review (Jason, 2026-09-30):**
  - **One bar per day.** The `ohlcv` key includes the source, so the 46 CoinGecko BTC/ETH days already stored sit next to Alpaca's bars for the same days. `recent_daily_candles` now returns one bar per day, with Alpaca's winning (`PREFERRED_CANDLE_SOURCE`); the 200-day line counts bars as days. The CoinGecko rows stay in the table.
  - **A refused cutover sell is retried until the timeout.** Alpaca cancels in the background, and a stop still pending its cancel holds the shares. A venue failure now ends the CLI with its message, not a traceback.
  - The cutover's order key reaches only the paper venue: an Alpaca sell closes the position and takes no key. A same-day re-run is safe because a sold position is no longer listed.
- CoinGecko is now called only without Alpaca keys (tests, a fresh clone), and the Fear & Greed index has its own client.

## D-047 - A Discord alert when the whole account is 15% below its peak; alert only (Jason, 2026-10-01)

**Decision (Jason, 2026-10-01: "Yes add a Discord alert"):** when the whole account first falls 15% (`ACCOUNT_DRAWDOWN_ALERT_PCT`) below its deposit-adjusted peak, the risk step logs the error-level event `account_drawdown_alert`, and the notifier posts it to Discord. Nothing is paused.

**Why:**
- Since D-045 the breakers and the drawdown kill switch watch the empty swing sleeve, so nothing watched the whole account. The backtest's worst time-weighted drawdown was 29.8% (tune half).
- Alert only, because stopping trading would stop the deposit buys during the fall (D-039: halting core buying in ordinary corrections collapsed returns) and the core's 25% trail sells, the one automatic exit left.

**Consequences:**
- `risk_manager._alert_account_drawdown` compares this cycle's drawdown with the previous cycle's snapshot: it pages once on the crossing, stays quiet while the account stays below, and re-arms after it recovers above 15%. A cycle hovering at the line can page on each crossing.
- It never fails the risk step: an error in the check logs `account_drawdown_alert_failed` as a warning.
- RISK_RULES.md is unchanged: the alert adds no trading rule.

## D-048 - Swing trading restored: D-045 reverted, back to D-040's 70/30 (Jason, 2026-10-02)

**Decision (Jason, 2026-10-02: "bring it back"):** the swing sleeve trades again.
- **The swing code was idle.** The sleeve's view, index and breakers, the analyst, the trend filter, the news and ML jobs all ran on nothing under D-045.
- **Targets:** back to D-040: 70% core (VOO 37.5%, QQQM 22.5%, BTC 6%, ETH 4%) and 30% swing.
- **Watchlist:** back to VOO, QQQM + the 13 swing stocks.
- **Rules:** the swing sleeve trades under D-042's rules, unchanged: a 25% cap of swing equity, a 20% reserve inside the sleeve, 1% risk of the account, whole shares, and breakers and kill switch on the sleeve's own index.
- **Not restored:** the proposed cash sweep (parking idle swing cash), whose live build was never merged; the reserve stays.

**Code:** D-045's code and test changes are reversed exactly (`core.CORE_TARGETS`, the default watchlist, `.env.example`, the backtest's report lines, and the tests' `swing_funded` fixtures). The later changes that assumed an empty sleeve are re-worded, and the empty-sleeve test becomes a 30%-of-deposits test. RISK_RULES.md keeps D-040's core rules and drops D-045's wording.

**Evidence, unchanged, and against it (past results only):**
- No backtest showed the swing sleeve adding return (D-039, D-045).
- The swing-never-trades run matched the best result in both halves.
- The 8-week paper period is now also the test of whether Claude's veto on entries beats the plain trend rule.

**Small-account limit (D-042), measured 2026-10-02 on the 2026-09-28 closes:**
- The sleeve starts at $300, and a 25% cap is $75 per position. The cheapest swing stock is WMT at $108.74, so **no swing stock is affordable in whole shares at the start.**
- The sleeve grows by 30% of each deposit. WMT fits after the first deposit and XOM (~$163) after about three. NVDA and AMZN (~$230-246) fit after about five; AAPL, GOOGL, JPM, AVGO and V (~$337-368) after about eight. MSFT, META, COST and LLY ($509-1,186) take 12-30 deposits.
- Until then the analyst proposes, and the gate refuses `position_size_zero` (counted in the weekly report, never paged). Claude is called every in-session cycle again.

**Timing:** the one-time switchover task pulls `main` at 09:35 ET on 2026-10-02.
- Merged before that, the ledger starts on 70/30.
- Merged after a 100%-core start, the core would already hold the swing share in VOO. `stock_core_cash` would then go negative, and swing cash would count about $300 that sits in VOO. Postpone the switchover rather than merge late.

## D-051 - The core + swing backtest runs the live core rules and models each venue's cash (2026-10-02)

**Decision:** Fix 1 of the pre-paper-clock fixes. `backtest/core_swing.py` no longer keeps its own copy of the core strategy.
- `portfolio/core.py` gains four pure functions, and live `plan()` and the backtest both call them:
  - `average_closes` (the 200-day line, used by `crossed_back`);
  - `trail_breached` (the trail check, pulled out of `plan()`);
  - `budgets` (what each venue may spend);
  - `venue_buys` (each venue's budget spread over its own symbols with `allocate`).
- The backtest's `core_allocation`, `trend_line`, its $1 `MIN_ORDER` and its own re-entry bookkeeping are deleted.
- **Each venue keeps its own cash, as live:** the stock venue gets 90% of each deposit (the core's 60% + swing's 30%), the crypto venue 10%. Core sells return to their own venue, and nothing crosses between venues.
- The stop alternatives the D-040 study compares (no stop, 3x ATR, 15% trail, below the 200-day line) stay as study-only rules; only the trail rows go through `trail_breached`.
- **No strategy constant or risk limit changed.**

**What the copies got wrong:**
- **Crypto over its cap:** the backtest moved crypto's capped share into VOO/QQQM. Live, the crypto venue's cash cannot reach stocks; it waits. A test pinned the divergence before the change: the same over-cap state bought $950 of stocks in the backtest and $891 live. It now asserts the two agree.
- **Small buys:** the backtest's minimum order was $1; live waits for $10.
- **Leftover cash:** the backtest also spread cash beyond every shortfall by target weight, and counted a close at or below the 200-day line on the stop's own decision day toward re-entry. Live counts from the day the sell filled.

**Old vs new** (`python -m trading_agent.backtest.core_swing` and `--sweep` on 2026-10-02, the same history for both; tune 2021-10-01 to 2024-03-31, validate 2024-04-01 to 2026-10-01; money-weighted yearly return / time-weighted max drawdown; past results only). These differ from the figures recorded in D-039 to D-044 because the history now runs a week further.

D-040 study, kill switch on the swing sleeve:

| Core stop | Tune old | Tune new | Validate old | Validate new |
|---|---|---|---|---|
| no stop | 21.5% / 27.5% | 20.2% / 27.3% | 13.3% / 20.0% | 12.5% / 20.0% |
| 3x ATR | 20.3% / 22.2% | 20.3% / 21.4% | 8.5% / 12.9% | 9.3% / 11.6% |
| **trail 25% (live)** | 19.0% / 22.8% | 18.9% / 22.7% | 13.6% / 18.3% | 13.2% / 18.2% |
| below 200-day | 17.8% / 18.1% | 17.2% / 17.6% | 11.5% / 12.5% | 10.1% / 12.3% |
| trail 15% | 14.0% / 18.7% | 13.7% / 18.7% | 10.1% / 16.9% | 9.8% / 16.8% |

D-044 sweep study (25% trail):

| Config | Tune old | Tune new | Validate old | Validate new |
|---|---|---|---|---|
| D-039 account breakers, 20% reserve | 19.0% / 22.8% | 18.9% / 22.7% | 13.6% / 18.3% | 13.2% / 18.2% |
| D-042 sleeve breakers, 20% reserve (live) | 22.3% / 24.5% | 22.2% / 24.4% | 13.8% / 18.2% | 13.4% / 18.1% |
| D-043 VOO sweep, option (a) | 23.6% / 25.9% | 23.5% / 25.8% | 18.9% / 20.4% | 18.5% / 20.4% |
| D-043 option (b) | 23.1% / 27.7% | 22.9% / 27.5% | 17.3% / 20.8% | 16.9% / 20.7% |
| Every deposit into VOO | 16.4% / 25.1% | same | 17.0% / 19.0% | same |

**Findings:**
- Every core configuration loses 0.1 to 1.4 points a year. The largest drops are no stop and below 200-day, whose idle crypto cash now waits instead of buying stocks.
- 3x ATR now edges no stop on the tune half (20.3% vs 20.2%), a tie within noise. On validate, the 25% trail is still the best stop rule (13.2%), so D-040's choice stands.
- D-044's conclusions stand: (a) beats (b) on validate (18.5% vs 16.9%). The sweep is the only configuration above every-deposit-into-VOO on validate.
- The live rules (D-042 with a 20% reserve) return 13.4% on validate against VOO's 17.0%.

**Consequences:**
- Changing the core's rules in `portfolio/core.py` now changes the backtest with them.
- One test, `test_without_a_ledger_a_core_symbol_is_reprotected_like_any_position`, fails on main and on this branch alike (`no_stop_computable` for VOO against the shared database). It is not caused by this change and is left for its own fix.

## D-053 - One halt policy function; the scheduler's pauses stop new entries but run the core and its protection (2026-10-09; design option A chosen by Jason)

**Decision:** Fix 3 of the pre-paper-clock fixes, built to option A of the Fix 3 design note (Jason, 2026-10-09). It amends D-019 (2): the scheduler's own pauses no longer skip the cycle.
- **One policy function.** `risk.killswitch.stops_everything(halted, reason)` is asked by the scheduler, risk node and executor. It replaces `protective_orders_allowed` and the scheduler's `skips_cycle`. The design note described three outcomes (normal, entries stopped, everything stopped); no caller tells the first two apart, so the function answers the one question they ask.
  - **Stops everything (True):** the manual kill switch, an unreachable Redis, and any reason the function does not know. The scheduler skips the cycle.
  - **Stops only swing orders (False under a halt):** any cooldown (the gate's breakers and the scheduler's own crash and connectivity pauses) and the automatic drawdown kill switch. Swing orders stop, new entries and trail closes alike (swing positions keep their venue stops); the core's orders, stop re-protection, snapshots and fill reconciliation continue.
- **The scheduler's pauses no longer skip the cycle.** Before, a crash or connectivity pause skipped every cycle for an hour, so the core's 25% trail was never checked and missing stops were never re-placed. A crashed cycle is now resumed on the next run, 15 minutes later instead of an hour.
- **Pages, as the design note says: when a pause starts, not every cycle.**
  - A pause is not re-armed or re-announced while it is active, checked in Redis at the moment of deciding (a pause that lifted mid-cycle is armed again). It lifts after its hour, and a problem that persists starts a new one with a new page: hourly at most, as before.
  - During a pause, a kind of error pages only the first time (`graph/nodes/notifier.hold_repeats`); its repeats are logged as `error_pages_held`. The page that starts the pause seeds that record, so a venue outage's `order_lookup_failed` and a repeat crash's `cycle_crashed` page once, not every cycle. Fills, closes and stop breaches still post.
  - A new kind of error during a pause (for example `account_drawdown_alert` or `order_lost_at_venue`, which fire only once) pages at once. Nothing is delayed or dropped; only repeats are held.
- **The pause reasons live in `risk/killswitch.py`** (`RUNNER_PAUSE_REASONS`, `scheduler_paused`) so the notifier node can ask about them without importing the scheduler.

**Evidence (2026-10-09 20:04 UTC, after the close; the live agent stopped for the run and restarted at 20:05):** one real paper cycle through `AgentRunner.run_once` with `cooldown:cycle_crashed` set in Redis.
- `run_once` returned `ran`; before this change it returned `skipped_halted` and nothing ran.
- Logs: `analyst_skipped` (no LLM spend), `execution_halted cooldown:cycle_crashed`, `cycle_summary`.
- Decisions row of cycle `e21a3239-8359-4455-9789-1edf4b3267a3`: the core was planned (`audit.core`: no buys, because only $5.91 of stock and $0.21 of crypto cash were uninvested), not `{"halted": ...}`.
- Stocks were closed, so no stock prices: no trail could fire and no re-protect was due. A core trail sale (VOO) and a re-protect under the pause are shown end to end by `test_under_the_schedulers_crash_pause_the_core_and_its_protection_still_run`.

**Consequences:** the gate's own breaker-reason mapping and the kill switch's identity checks (`/killswitch` read-back, the gate re-arming the drawdown switch) are not policy and stay as they are.

## D-054 - Fractional swing positions; the app holds their stop (Jason, 2026-10-10)

**Decision (Jason, 2026-10-10):**
1. **Option A: the app holds the stop for swing positions.** No stop order sits at Alpaca for them.
2. **Minimum swing order: $10**, the same as the core's `MIN_ORDER` (`RiskLimits.min_order_value`).
3. **An approved exception to the strategy freeze.** The swing sleeve made no trades since the 2026-10-02 cutover (every buy was rejected `position_size_zero`, "under one whole share"), so its measurement starts when this merges. The core's 8-week clock keeps running.
4. **App-held stop sales are protective, like the core's trail sales.** They go out whenever `risk.killswitch.stops_everything` is False (every cooldown, the scheduler's pauses, the automatic drawdown switch), and stop only when it is True (the manual kill switch, Redis unreachable). This amends D-053, which held swing orders under halts on the assumption that swing positions had venue stops. New swing entries still stop under every halt.

**Step 0 (verified 2026-10-09 on the paper account and Alpaca's docs):**
- Alpaca accepts a fractional stop order only as **DAY**. A GTC one is refused ("stop/stop_limit fractional GTC orders are not enabled"), and a fractional buy with an attached stop is refused ("fractional orders must be simple orders").
- A sell stop above the market is refused ("stop price must be less than current price").
- All 13 swing stocks are fractionable.
- Jason chose option B (a daily stop at Alpaca) on 2026-10-09, then option A on 2026-10-10 with these facts known.

**What changes:**
- **`risk/gate.py`:** no whole-share rounding. A stock buy under $10 of notional is rejected `position_size_zero` with the amount in its detail (crypto keeps its venue's own minimum). Sizing and the ATR stop are unchanged; the stop stays on `ApprovedOrder.stop_loss_price`. A buy the position cap shrinks below $10 is rejected `position_size_zero`, not `position_cap_reached`. `core.MIN_ORDER` now reads the same $10 (`RiskLimits.min_order_value`).
- **`brokers/alpaca.py`:** every buy is a fractional DAY market order with no stop attached. The OTO path is deleted.
- **The app's stop:**
  - It is the `stop_loss_price` on the position's latest market buy row since the symbol's last filled sale (`OrderRepository.entry_stop`), booked by the executor as before. A previous position's buy never counts; a buy retired as lost still does.
  - Each cycle `PortfolioTracker.check_swing_stops` compares this cycle's price with it for every non-core position. A price at or below the stop is a `StopBreach`, which the executor sells in full.
  - The check runs right after the portfolio is assembled, before the sleeve is measured. If a later part of the risk step fails, the breaches already found are still handed to the executor and sold.
  - A position with no price, or no stored stop, is skipped with a warning and never treated as safe or breached; a missing stop is also reported once at error level.
- **`graph/nodes/trade_executor.py`:**
  - One function, `_goes_out`, decides what a halt holds back, and re-protect uses the same answer for which symbols are being closed. Every stop-breach sale is protective, the fixed trail's too if it is ever switched back on (D-032); one sale per symbol, with client order id `<cycle>:<symbol>:stop`.
  - Re-protect never places a venue stop for a position whose stop the app holds.
- **`backtest/core_swing.py`:** swing buys fill fractionally. A buy the cash cannot fund above the minimum is counted as unaffordable. Its stop is still checked against each day's low, which is optimistic: live checks it every 15 minutes during the session and not at all while the agent is down.

**Known limits:**
- The app checks the stop once per 15-minute cycle and only when the symbol has a price (the NYSE session). A stop is therefore sold up to 15 minutes late, at that cycle's price. After an overnight gap it sells at the first in-session price.
- **There is no stop at all while the agent is not running** (the host machine off or asleep, D-003). A venue stop would have kept working; that was option B, and Jason chose A.
- **A buy whose order row was never written** (`submission_unbooked`, which pages) has no stored stop. The tracker reports it once at error level, and re-protect cannot place a whole-share venue stop on a fraction. A human fixes the book.
- **A position with no price is skipped with a warning, not a page.** If the price feed fails for the whole session, every swing stop goes unchecked until it returns.
- **A triggered stop is not remembered.** If its sale fails (that pages) and the price is back above the stop next cycle, the position is kept; a venue stop would have sold.
- **A same-day stop sale is a day trade.** Under $25k, Alpaca's pattern-day-trader check could refuse a fourth in five business days, leaving the position until the next session.
- **If the portfolio cannot be assembled at all** (a broker or database outage before `build_state` returns), no stop is checked that cycle; the scheduler pauses and pages (D-019, D-053).

**Backtest** (`python -m trading_agent.backtest.core_swing` and `--sweep`, 2026-10-10; before = main, after = this change, the same history: tune 2021-10-01 to 2024-04-04, validate 2024-04-05 to 2026-10-09; money-weighted yearly return / time-weighted max drawdown; swing = the sleeve's own money-weighted return; past results only):

| Config | Half | Before | After |
|---|---|---|---|
| **D-042, the live rules** (sleeve breakers, 20% reserve, 25% trail) | tune | 20.2% / 24.4%, swing 14.2%, 133 trades, no kill switch | **14.7% / 27.4%, swing -4.6%**, 56 trades, kill switch 2022-08-26 |
| | validate | 15.8% / 18.1%, swing 6.4%, 134 trades, no kill switch | **13.5% / 18.0%, swing -2.2%**, 152 trades, kill switch 2026-06-26 |
| D-043 VOO sweep, option (a) | tune | 21.2% / 25.8%, swing 17.7% | 20.4% / 26.6%, swing 15.0% |
| | validate | 19.3% / 20.5%, swing 18.4% | 19.0% / 20.6%, swing 17.2% |
| D-040 study, trail 25%, swing-only kill switch | tune | 17.1% / 22.7% | 15.8% / 24.2% |
| | validate | 13.6% / 18.0% | 13.7% / 17.9% |
| Every deposit into VOO | tune / validate | 14.6% / 25.1% · 18.4% / 18.9% | same |

- **Under the live rules the swing sleeve does worse fractional.** With whole shares the small early sleeve could rarely buy, so its cash sat out the 2022 fall. Fractional buys trade from the first deposit, the early losses trip the sleeve's kill switch, and the plain trend rule then stays off until a human `/resume` the backtest cannot model.
- The backtest runs the plain trend rule, not Claude. The paper period measures Claude's veto, which is the point of this exception (decision 3).

## D-057 - Claude sees which swing stocks the account holds, never how many (2026-10-11)

**Decision:** amends D-038 (2): "propose sell when trend_up is 0" now applies to held stocks only.
- **The fetch step reads every venue's positions** into `market_data["held"]`, the sorted symbols with a positive quantity (the gate's definition of held). If any venue's read fails, `held` is left out.
- **The analyst's snapshot gets `held`:** the held symbols among the priced swing symbols, so a core holding never appears. Names only, never quantities: the LLM still never sees sizes (RISK_RULES structural enforcement), and a sell is always a full close, so a count could not change a decision.
- **The prompt rules:** sell a symbol in `held` whose trend_up is 0, and never any other; buy only a symbol not in `held`; otherwise hold.
- **No holdings, no Claude call.** When `held` is missing the analyst skips the cycle (`analyst_skipped reason="holdings unknown"`). Without holdings it would miss real exits, and the risk step reading the same venues would most likely fail closed anyway.
- **No analyst-side filter for unheld sells.** The gate's `no_position_to_sell` already rejects them. One check in one place, and the gate's count stays an honest measure of the prompt fix.

**Why:** the prompt told Claude to sell every stock below its trend "if one is held", but its input had no holdings, so it proposed a sell of every downtrend watchlist stock every 15 minutes: 1,080 `no_position_to_sell` rejections from 2026-10-02 to 2026-10-09 (8 symbols, 8 per cycle, 208 per full trading day).

**Consequences:**
- On the paper broker, `get_positions` settles crossed stops. That now happens in the fetch step, before the risk step reconciles orders, so a paper stop fill is booked in the cycle it settles instead of the next one (15 minutes sooner; the amounts are unchanged).
- Each venue's positions are read twice a cycle (fetch step, then the tracker): one more venue call per cycle.
- The full-day after count needs the change deployed to the running agent.

## D-059 - CI runs the machine checks on every PR (2026-10-15)

**Decision:** CI for every pull request.
- **`.github/workflows/ci.yml` runs on every PR push:** `ruff check`, `ruff format --check`, the full pytest suite and `scripts/pr_size.py`. The tests run against TimescaleDB (`latest-pg16`) and Redis 7 services with docker-compose.yml's credentials, so the default `DATABASE_URL` and `REDIS_URL` work. No secrets: the two `test_worker.py` news tests stub Alpaca keys instead of reading a local `.env`, so no test needs skipping.
- **Size limits.** `scripts/pr_size.py` counts added and removed code lines. Blank lines, comment lines and Python docstrings (found with `ast` on both file versions) don't count, and `.md`, `.lock`, `.json` and `.txt` files are skipped. Over 800 it warns; over 1200 it fails; in between the size is a judgment call.

**Why:** every check depended on an agent running it by hand, and a worktree without a `.env` failed two tests, so "green" meant different things on different machines.

**Cost:** about 3-4 Actions minutes a run, well inside GitHub Free's 2,000 private-repo minutes a month. Cancel-in-progress and a 15-minute timeout cap waste; with the Actions budget at $0, jobs stop at the limit instead of billing.

**Consequences:** `test_shipped_calendar_is_not_about_to_run_out` goes red around 2026-11-04 until the 2027 BLS dates are added, and from then on it fails every PR's CI.

## D-060 - A sell stands only when its trend_up is 0, and Claude's reasoning is stored (Jason, 2026-10-17)

**Decision (Jason approved, 2026-10-17):** an approved exception to the strategy freeze, amending D-057's prompt-only sell rule.
- **The analyst discards a sell whose trend_up is not 0**, as it already discards a buy whose trend_up is not 1 (D-038). With no trend label (under 50 complete daily bars) there is no trend exit; the stop still protects. Dropped sells are logged as `analyst_sell_against_trend_dropped` with symbol, trend_up, confidence and reasoning.
- **Unheld sells are still left to the gate** (D-057): a downtrend sell of a stock not held still reaches `no_position_to_sell`.
- **Each proposal's reasoning is stored** in the risk_manager decisions row next to symbol, action and confidence.

**Why:** on 2026-10-16, the first live day of fractional swing buys (D-054), Claude sold held GOOGL (13:30 ET) and META (13:45 ET) while both had trend_up 1. The prompt says never to do that, but nothing in code enforced it. GOOGL was bought back 15 minutes later. Realized −$2.14 and −$3.97, plus two same-day round trips that count as day trades. Its reasoning was never stored, so the sells could not be explained afterwards.

**Consequences:** the trend exit becomes deterministic, matching the backtested rule. Claude can no longer exit early on a held uptrend stock, even with a good reason, so only the stop or the trend turning down closes a swing position.

## D-061 - Each deposit records its core part when booked, and a swing failure stops swing only (Jason, 2026-10-11)

**Decision (Jason, 2026-10-11, confirmed 2026-10-17):** option A of the design note, and the $1.03 cutover fee stays in swing history (no adjustment row).
- **The split is recorded, not recomputed.** Each `cash_flows` row stores `core_amount` when it is booked: all of a crypto row, `ledger.CORE_STOCK_SHARE` (60%) of the whole deposit on the stock row, 0 on a baseline. A deposit books 90% at the stock venue (core 60%, swing 30%) and 10% at the crypto venue (all core). `core.stock_core_cash` and the swing index's new money read these amounts, and `CORE_STOCK_SHARE` is used only when booking (`ledger.core_split`, shared with the backtest). It moved from `core.py` to `ledger.py`, which books the rows; a test keeps the core's VOO and QQQM targets equal to it.
- **Rows booked before the column:** `init_db` adds `core_amount` and fills in any row without one with the split in force before it (the 2026-10-02 and 2026-10-16 deposits). No money moves. The column stays nullable on an existing database, so an agent still running the old code can book a deposit and the next `init_db` fills it in; a fresh database has it NOT NULL.
- **Swing cash still reads the venue.** It is the stock venue's cash minus the core's recorded claim, so anything the ledger and the booked fills don't explain (a venue fee; a dividend, if one ever posts) lands in swing by rule. Each cycle's `audit.swing.unexplained` shows that amount: the stock venue's app cash minus the stock deposits and every stock fill since the start. A fill the book missed can therefore never fake a swing gain, which option B could not rule out.
- **A swing failure stops swing only.** Measuring the swing sleeve and the swing gate get their own `try` in the risk step. When either raises:
  - `swing_measure_failed` is logged at error level (Discord pages);
  - every swing proposal is rejected `evaluation_error` ("swing sleeve not measured: <type>");
  - the snapshot is written without swing figures;
  - the core's orders and every stop sale still go out.

  `_swing_view` no longer writes the snapshot: the risk step writes it after the gate, with the swing figures only when they were measured. A failure in the shared inputs (reconcile, `build_state`, the core's book) still fails the whole step closed, with stop sales found before it still sent (D-054).

**Step 0:** Alpaca paper accounts don't process corporate actions, dividends included (Alpaca staff on its forum, https://forum.alpaca.markets/t/corporate-actions-handeling/19325). So on paper only fees reach `unexplained`; in live trading dividends would too. That is when the dividend part of option C becomes worth building, together with Phase 4's deposit detection (D-040).

**Why:** a change to the 60% constant would have moved money deposited in the past, and a bug in measuring the swing sleeve stopped the core's deposit buys and trail sales for that cycle.

**Evidence plan (tier 2):** one real paper cycle after Jason deploys. Its decisions row should show `audit.swing.core_claim` read from the stored amounts and `audit.swing.unexplained` at about -$1.03.

## D-062 - Hold the swing strategy steady through the paper period (Jason, 2026-10-17)

**Decision (Jason, 2026-10-17):** "We hold the swing strategy steady for a while. We want to observe these changes over a longer period of time. Each rule change during the paper period resets the 8 week measuring results."
- The swing rules in force after D-060 (fractional buys with app-held stops, D-054; sells only when `trend_up` is 0, D-060) are what the paper period measures. Its 8 weeks count from when the last of them is running.
- The strategy freeze applies with no further exceptions: fixes only, unless Jason says otherwise. A problem with the strategy found meanwhile is written down for Jason, not changed in a rule.
- Not covered: bug fixes and accounting fixes that change no rule, such as Fix 2 (D-061), which moves no money and changes no limit.

**Why:** D-054 (2026-10-14) and D-060 (2026-10-17) each changed the swing rules within days of each other. Every change restarts the question the paper period answers, whether Claude's veto beats the plain trend rule, so it can't be answered while the rules keep moving.

## D-064 - The spec: two finish lines, a 4-week paper test and a +5% swing bar (Jason, 2026-10-18)

**Decision (Jason, 2026-10-18):** `docs/SPEC.md` defines what done means and the bar before live money. It amends D-030's staged goal and the 8-week paper period in D-062.
- **Goal:** trade live money for Jason. 70% index funds and safe investments, 30% swing budget. No day trading, no leverage, no promised returns.
- **Two finish lines:** functional done (SPEC F1–F9, U1–U5), then the live gate (L1–L6) plus Jason's explicit go.
- **Paper test:** 4 weeks with no rule changes, 2026-10-19 to 2026-11-13, replacing the 8 weeks.
- **Swing bar:** each swing trade is judged on its own investment, after fees. The dollar-weighted total must be at least +5% over the window, with no minimum trade count. VOO is reported for analysis only and is not a requirement.
- **First live scope:** stocks only on Alpaca, with a small amount. The live core is VOO and QQQM, which take the core's 10% crypto share until crypto rules are tested.
- **Accepted extras:** the guardrails (L4); a live review every 2 weeks, where a miss returns to paper (L6); and the missed-cycle and unchecked-stop alerts (U3, U4), both required before live.
- **Later:** options and shorting inside the 30%, defined risk only (bought calls and puts; no selling options, no borrowed shares, no margin), as its own stage with its own paper test. Crypto rules are tested separately.

**Why:** there was no spec. The docs' finish line (beat buy-and-hold) and Jason's (trade 70/30 on its own) disagreed, so each PR picked its own, and strategy exceptions were made case by case. Jason drafted the spec on 2026-10-18 and set every number in it.

**Consequences:**
- The work the spec requires before live is queued: U3, U4 and the weekly report's swing-bar and VOO views (F8).
- Jason confirmed on 2026-10-18 that a window with no swing trades is extended by two weeks (L3), and that the 2-weekly live review checks the last 4 weeks against the +5% bar (L6).

## D-071 - Cost basis counts from the ledger start, and the broker referees the app's positions (Jason, 2026-11-09)

**Decision (Jason, 2026-11-09):** "go with A, two in a row, you run write".
- **The rule:** a symbol's cost basis uses only its fills from the paper ledger's start on, as the ledger already says it does. `TradeRepository.fills_for_symbol` takes `since`, and the executor's realized P&L, the tracker's fallback entry and its trailing-stop entry time pass the ledger start (`CashFlowRepository.start`). Without a ledger nothing changes. The core's trail already worked this way.
- **The referee:** each cycle, after the book is reconciled, the tracker compares the app's open quantity per symbol (its fills since the ledger start) with the broker's. A difference over 0.000001 units that lasts two cycles in a row logs `position_mismatch` at error level, which pages Discord. It pages once per symbol until the quantities match again (`position_mismatch_cleared`). It is read-only, is skipped while a venue's book failed to reconcile, and a failure inside it is logged as a warning. The warning `cost_basis_quantity_mismatch`, which covered part of this on one path, is deleted.
- **The correction:** `python -m trading_agent.portfolio.repair_pnl` recomputes realized P&L for every paper closing fill since the ledger start. It lists the changes as a dry run, and `--write` writes them, logging each old value first so the write can be undone.

**Why:** the 2026-10-02 cutover sold AAPL 32, MSFT 21, NVDA 24, SPY 14 and XOM 68 without booking the sales (by design, `ledger.close_positions`). The cost-basis walk read every fill ever booked, so those lots still looked open and blended into later buys. AAPL's 2026-11-04 swing sale was booked at +2.28; its own result was -1.19. Option B (book the cutover sales as trades) was rejected: the next cutover would repeat the bug.

**Consequences:**
- Realized P&L feeds the consecutive-losses breaker as well as the weekly report and SPEC L2. Correcting AAPL turns a counted win into a loss: the swing sleeve's streak goes from 0 to 1, far below the 5-loss pause. This changes what the practice run's gate saw, but no rule changed.
- The fix and the referee need an agent restart (Jason's step), off-hours before 2026-12-04, together with the 2027 macro calendar. Until then the running agent books sales the old way (MSFT's 21 cutover shares still blend into its swing lot), so `repair_pnl` runs again after the restart, dry run then `--write`; a rerun corrects only rows still booked under the old rule.
