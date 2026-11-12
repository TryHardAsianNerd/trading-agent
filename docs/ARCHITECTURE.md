# Architecture

Approved by Jason on 2026-09-02. Change only via a new entry in [DECISIONS.md](DECISIONS.md).

## Overview

A single long-running Python service, orchestrated as a LangGraph state graph, running locally in Docker Compose. APScheduler triggers an agent cycle every 15 minutes (market-hours-aware for stocks; 24/7 for crypto):

```
fetch → features → predict → analyze (Claude) → risk gate → execute → notify
```

Graph state is checkpointed to Postgres so the agent survives restarts with portfolio context intact.

## Execution model by asset class

| Asset | Autonomy | Venue |
|---|---|---|
| US stocks | Fully autonomous | Alpaca (paper → live, same API) |
| Crypto | Fully autonomous | Coinbase Advanced Trade (sandbox → live) |

## Repo layout

```
├── docker-compose.yml          # timescaledb, redis, worker (D-025)
├── Dockerfile                  # worker image only; the agent runs on the host
├── .env.example                # all keys documented, never committed
├── pyproject.toml
├── src/trading_agent/
│   ├── main.py                 # entry point
│   ├── config.py               # pydantic-settings; TRADING_MODE flag
│   ├── graph/
│   │   ├── state.py            # AgentState (portfolio, signals, decisions)
│   │   ├── builder.py          # wires nodes into the StateGraph
│   │   └── nodes/
│   │       ├── data_fetcher.py     # Finnhub (stocks) + CoinGecko (crypto)
│   │       ├── feature_engine.py   # pandas-ta Tier 1 indicators
│   │       ├── ml_predictor.py     # XGBoost directional model; logged, not shown to the analyst (D-032)
│   │       ├── llm_analyst.py      # Claude via langchain-anthropic
│   │       ├── risk_manager.py     # deterministic gate (calls risk/)
│   │       ├── trade_executor.py   # Alpaca paper / Coinbase sandbox
│   │       └── notifier.py         # Discord (D-017)
│   ├── brokers/                # alpaca.py, coinbase.py behind one Broker interface
│   ├── data/                   # provider clients + TimescaleDB writers
│   ├── ml/                     # directional model (D-024); FinBERT news sentiment, worker-only (D-026)
│   ├── backtest/               # daily replay of the model through the live risk gate (D-029)
│   ├── risk/                   # the 5 risk layers as pure, unit-testable functions
│   ├── storage/                # SQLAlchemy models, repositories
│   ├── notify/                 # discord bot: alerts + /killswitch, /resume, /status commands
│   ├── scheduler.py            # APScheduler cycle trigger
│   └── worker.py               # Celery app + scheduled background jobs (D-025)
└── tests/
```

## Storage

One database: **TimescaleDB** (it IS Postgres) serves both OHLCV hypertables and relational tables (trades, orders, portfolio snapshots, decision log). **Redis** holds the kill-switch flag, circuit-breaker state (with TTL cooldowns), position cache, and rate-limit counters. No SQLite anywhere.

## Design principles (hold firm)

1. **Risk manager is code, not AI.** The five risk layers ([RISK_RULES.md](RISK_RULES.md)) are pure deterministic functions in `risk/` with hard-coded limits. The Claude analyst proposes; the risk gate disposes. LLM output can never see or modify risk parameters.
2. **Kill switch in Redis, checked everywhere.** `/killswitch` (Discord) sets the flag; scheduler, executor, and risk gate each check it independently; setting it cancels open orders. Circuit breakers set the same flag with TTL.
3. **Paper/live is one config flag with friction.** `TRADING_MODE=paper` default; this public build is paper-only and rejects live mode outright. Same code path throughout.
4. **Every decision is auditable.** structlog JSON records inputs, model outputs, risk-gate verdict, and order result for every cycle.

## Deliberate deviations from the blueprint (Phase 1)

- **Celery arrived in Phase 2 for background jobs only** (D-025) — a `worker` container runs slow jobs (the weekly model retrain; FinBERT news scoring every 10 minutes, D-026) so the 15-minute cycle never waits on them. The trading cycle itself stays on APScheduler in the host process.
- **No SQLite** — TimescaleDB from day one covers everything.
- **ML predictor was a stub in Phase 1** — graph shape was real from day one; XGBoost slotted in without rewiring (D-024). Its predictions are logged for evaluation; since D-032 they reach neither the LLM analyst nor the risk gate.
- **No FastAPI surface yet** — Discord + logs are the whole interface for now (D-017; Telegram before that). FastAPI arrives when a dashboard is wanted.

## External services & keys (all free tier for Phase 1)

| Service | Purpose | Env var |
|---|---|---|
| Finnhub | Stock quotes (60 req/min; free plan has no candles) | `FINNHUB_API_KEY` |
| CoinGecko | Crypto market data (30 req/min, no key needed on free tier) | — |
| Alpaca | Stock paper trading + daily bars (IEX feed, D-021); ML training history for stocks + crypto (D-024); news headlines for sentiment (D-026) | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |
| Coinbase Advanced Trade | Crypto sandbox | `COINBASE_API_KEY`, `COINBASE_API_SECRET` |
| alternative.me | Crypto Fear & Greed Index, daily, for crypto features (D-027) | — |
| FRED (St. Louis Fed) | VIX daily close for the Layer-5 environment (D-023); key travels in the URL | `FRED_API_KEY` |
| Anthropic | Claude analyst node | `ANTHROPIC_API_KEY` |
| Discord | Alerts + control commands (D-017) | `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`, `DISCORD_OWNER_ID` |

## Future (out of scope now, don't build for it yet)

Phase 2 (done): XGBoost, FinBERT sentiment, Fear & Greed, full risk-layer wiring; Polymarket dropped. Phase 3: backtesting (event-driven through the live gate, not VectorBT: D-029), months of paper trading. Phase 4: live with a small amount. Possible productization someday (signals-subscription first) — noted in DECISIONS.md, not an architectural driver today.
