# Sentinel Trader

## Gold-only release (current)

[Sentinel live](https://sentinel-trader-kp.streamlit.app/) now covers **gold only**. The home page (`gold_app.py`) shows:

- **Price ranges** for the next day, week and month at 50/80/90% confidence, plus how high or low tomorrow's price is likely to reach. These are the forecasts that held up on unseen data: on 2019–2026, the 80% range contained **78–80%** of outcomes and the 90% range **86–89%**, measured on non-overlapping periods. The volatility model (EWMA λ = 0.94 blended with gold's implied volatility, GVZ) and every setting were chosen on 2009–2018 only.
- **Direction tests, reported honestly:** walk-forward logistic regression, gradient boosting, and trend and momentum rules at 1/5/21/63 days. On 2019–2026 **none beat simply guessing "up"** (gold rose in 57–71% of periods). The page labels the direction signal "no reliable edge" and recomputes this verdict every time it loads.
- **What's moving gold:** the dollar, yields, real yields (TIP), VIX, silver, miners, oil and copper. **Psychology & demand:** fear & greed, RSI, GVZ, GLD ETF volume, streaks and drawdown. **Strategies:** buy & hold beat every rule on return; volatility-sized holding had the best Sharpe (0.91 vs 0.79) and the smaller drawdown (−26% vs −34%). **Seasonality.**
- A **trade planner** that sizes a position from the user's own stop, with stops placed using the calibrated high/low levels. Every level rescales to the user's broker XAU/USD quote.

**Final validation (21 Nov 2026)** is in [`research/gold`](research/gold/README.md) and on the page's *What we tested* tab:
- 40 tuned direction configurations, chosen on 2006–2018. Expanded drivers: CFTC hedge-fund positioning, credit stress, Fed policy, INR/CNY/JPY/EUR, Bitcoin, GVZ. All still lost to "always up" on 2019–2026.
- A HAR range-volatility upgrade gained only 0.2–4% and lost calibration, so the simpler model stays.
- The only "profitable" pattern (range breakouts) disappeared once fills were modelled realistically.
- CFD financing (~6%/yr) cut buy & hold from 13.7% to 9.0% a year.

### Daily gold plan (entry, stop-loss, exit)

The Gold page's **Daily plan** tab and banner apply one fixed rule:
- **Entry:** buy at the next open after a close above both the 50- and 200-day averages.
- **Stop-loss:** trails 2 daily volatilities below the best close.
- **Exit:** at the stop, or at the next open after a close below either average.

Track record with costs: 2008–2018 about flat; 2019–2026 +6.3%/yr; 37% winning trades; worst losing streak 8. Sized at 1% risk per trade: about +4–5%/yr with a −10% worst drawdown. Tuning 80 variants on 2006–2018 made 2019–2026 *worse* (rank correlation −0.33), so the rule is deliberately untuned (`research/gold/plan_tune.py`). Signals use completed sessions only; an in-progress daily bar is shown as the live price but never feeds the plan.

**Automate it on Alpaca paper (GLD):** `sentinel gold-plan --dry-run` shows what it would do. Each run:
- queues a risk-sized market buy for the next open, with an attached stop;
- raises (never lowers) a GTC trailing stop;
- or sells at the next open when the plan says exit.

On Windows, run `run_gold_plan.bat` once a day, or `schedule_gold_plan.bat` to schedule it every morning at 08:00 local time (after the US close).

The intraday paper bot (`config/config.yaml`) trades **GLD** only. The **Gold bot lab** page runs its 15-minute walk-forward backtest and forward simulation on GLD. Crypto, oil and the multi-asset analyzer are preserved in the repo but removed from the public navigation.

## LSTM, TFT, Random Forest and real-data paper challengers

The **LSTM · TFT · RF** tab adds 120 completed chronological fits across five usable instruments, with standard LSTM, actual PyTorch Forecasting TFT, Random Forest, LightGBM, Extra Trees, logistic regression, a prior and a regularized hybrid. Full candidate results and source/fit audits are retained. The new models did not establish broad superiority: the matched-date selected SOL baseline improved from 48.23% to 50.80%, BTC/ETH accuracy was unchanged, and the selected gold challenger was slightly worse. The deployed v3 inference models are preserved.

Separate local paper books now collect real BTC/ETH/SOL quotes and new forecasts, including observed spreads and assumed fees/slippage. Startup diagnostics are kept apart from the primary candle-close experiment. No new 24-hour outcome is claimed before it matures. See [methods, results and worker operations](crypto_lab/docs/CHALLENGERS.md). No real-money orders or exchange keys are used.

## Public research release v3

Open [Sentinel live](https://sentinel-trader-kp.streamlit.app/) for public BTC, ETH and SOL quotes, research probabilities, the complete accuracy comparison and commodity-data readiness. The website submits no trading orders. The original asset analyzer remains available in the sidebar.

The [v3 monthly-refit experiment](crypto_lab/accuracy_evidence/RESULTS.md) raised matched-date gold direction accuracy from **49.31% to 51.88%** and reduced Brier loss from 0.25540 to 0.24996. The descriptive improvement interval includes zero, and these dates were already inspected: this is exploratory evidence, not fresh proof of skill or profitability. BTC/ETH accuracy was unchanged; SOL declined from 48.40% to 48.08%. All candidate scores and uncertainty are published.

Gold and oil CFD live signals remain disabled until current broker-specific data and execution terms are validated. Hosted prices refresh while the page is open; the separate continuous paper processes run locally. See [deployment, operation and rollback](crypto_lab/docs/PUBLIC_RELEASE.md).

## Gold, USOIL-style CFDs and crypto focus

The [focused upgrade](crypto_lab/docs/FOCUS_RESEARCH.md) adds real XAUUSD and WTI reference minute-data ingestion, hourly session-aware CFD research, long/short accounting, causal market-context features, chronological probability calibration, bounded model tuning, broker bid/ask CSV import and a separate prospective challenger observer. Open **Gold & oil research** in the Streamlit sidebar or the first tab of the local monitor.

**The v2 tuning did not establish a general accuracy improvement.** Gold's v2 final retrospective direction accuracy was 49.3%; the selected ETH policy lost 1.17%. The WTI CFD reference source has a 940-day gap, so current CFD qualification is blocked. A separate daily `CL=F` direction study is clearly labeled and has no USOIL trading-P&L claim. No candidate is proven best in class or approved for broker execution. The original paper ledger is preserved.

See the [new results and limitations](crypto_lab/focus_evidence/RESULTS.md), [all model/strategy attempts](crypto_lab/focus_evidence/summary.json), and [broker-input template](crypto_lab/broker_profile.example.json). No passwords or exchange keys are required for these research workflows.

## Crypto research and continuous local paper simulation

The new [Crypto Lab](crypto_lab/README.md) tests BTC, ETH and SOL on 38,595 real four-hour candles. It includes 20 predeclared candidates, chronological model calibration, purged walk-forward validation, a frozen six-month test, fee/slippage stress tests, all 780 validation attempts, and a restartable local forward paper worker. Open **Crypto Research** in the Streamlit sidebar, or run its dedicated local monitor at port 8502.

**Current evidence: no profitable edge established.** The frozen test selected cash: 0.00% net, versus −15.70% for full-exposure buy & hold and −5.17% for the volatility-matched baseline. The separate forward experiment tracks actual future observations; its results must not be presented as historical earnings. This crypto package accepts no exchange keys and has no real-money order client.

See the [primary-source research review](crypto_lab/docs/RESEARCH.md), [sealed evidence](crypto_lab/evidence/summary.json), and [local start/stop instructions](crypto_lab/README.md). The existing US-equity code below is preserved; its historical performance claims are separate from this crypto experiment.

An intraday trading bot for US equities on [Alpaca](https://alpaca.markets). It combines a LightGBM probability model, chart-pattern features, a market-regime filter and real-time news sentiment, with an independent risk officer that can veto any trade or shut the bot down.

It runs in **paper trading by default**. Real money takes three separate switches (see [Safety](#safety)).

It also ships a **Streamlit app** (`streamlit_app.py`) for analysis and simulation:

- **Analyze any asset:** search any stock, ETF, crypto or commodity (e.g. `AAPL`, `BTC-USD`, `GC=F`, `RELIANCE.NS`). The app downloads up to 10 years of daily data and computes 49 factors: trend, momentum, mean reversion, volatility, **demand & volume** (OBV, accumulation/distribution, money flow), **psychology** (fear & greed composite, VIX, 52-week anchoring, drawdown, streaks), **macro** (market trend, rates, dollar, beta) and seasonality. It then runs 9 rule strategies, a walk-forward ML ensemble and an adaptive meta-strategy out-of-sample, and returns an evidence-weighted signal with its accuracy attached.
- **Accuracy Lab:** the same out-of-sample test across a basket of assets.
- **Trading bot lab:** the bot's 15-minute strategy on the last ~60 days of real bars. It shows the walk-forward backtest and a forward simulation in which a model that never saw the last N sessions trades them bar by bar.
- **Paper account:** a read-only view of your Alpaca paper account (keys go in Streamlit secrets).

```bash
pip install -r requirements.txt && streamlit run streamlit_app.py
```

## Results on real data (Yahoo Finance, to 13 Nov 2026)

These are reported as measured, not tuned to look good.

| Test | Result |
|---|---|
| Intraday bot, walk-forward, 10 US large caps, 15-min bars, ~33 sessions OOS | Model AUC 0.60. 49 trades, profit factor 1.23, **+0.5%** vs equal-weight buy & hold −6.5%, max drawdown −0.65% |
| Label/feature variants (TP/SL/horizon, market-context features), split-half check | The default (TP 2 ATR / SL 1 ATR / 8 bars, no market features) was best in both halves. Entry threshold 0.42 was the only one profitable in both halves |
| Daily direction, 5-day horizon, 17 assets, ~6 years OOS | Mean AUC **0.505**: indistinguishable from a coin flip. 3 of 17 assets p<0.05 (about what chance gives you) |
| Pooled cross-asset model, 5- and 21-day horizons | No improvement (AUC 0.47–0.50) |
| Rule strategies vs buy & hold (daily) | Trend and volatility filters mostly **cut drawdowns** (e.g. NVDA −66% → −12%). They rarely beat buy & hold on Sharpe |

Takeaway: the intraday bot shows a small edge that is not yet statistically proven. Daily direction prediction does not. The app weights each signal by its proven edge, so models with no edge get no vote.

```
sentinel fetch     ->  sentinel train  ->  sentinel backtest  ->  sentinel replay  ->  sentinel live
 (bars + news)        (walk-forward)       (out-of-sample)       (live code path)     (Alpaca paper)
```

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        WS1[Alpaca minute-bar websocket] --> AGG[N-minute bar aggregator]
        WS2[Alpaca news websocket] --> NLP
        REST[REST history: bars + news] --> DB[(SQLite audit trail)]
    end
    subgraph Signals
        AGG --> FE[Causal features + candlestick / double-bottom patterns]
        FE --> REG[GMM regime model]
        FE --> GBM[LightGBM: P take-profit before stop]
        NLP[FinBERT or lexicon, then optional Claude escalation] --> SENT[Time-decayed sentiment state]
        SENT --> GBM
    end
    REG --> FUSE{Signal fusion}
    GBM --> FUSE
    SENT -->|veto / boost / exit| FUSE
    FUSE --> RISK[Risk officer + vol-targeted fractional-Kelly sizing]
    RISK --> EXE[Bracket orders: TP / SL]
    EXE --> BRK[Alpaca or SimBroker]
    BRK --> DB --> DASH[Streamlit monitor]
```

The same `TradingEngine` drives backtests, replays and live trading. Only three things change between modes: where the data comes from, which signal provider is used, and which broker fills the orders. That means the decision logic, sizing and risk rules you backtest are exactly the code that trades.

| Layer | Module | What it does |
|---|---|---|
| Data | `data/alpaca_data.py`, `data/store.py` | Adjusted N-minute bars and news from Alpaca REST; SQLite (WAL) for market data and every decision, order, fill and risk event |
| Live ingestion | `live/runner.py`, `live/aggregator.py` | Websocket streams in threads → one asyncio queue; 1-min bars rolled into N-min bars aligned to the 9:30 open |
| Features | `features/technical.py`, `features/patterns.py` | 31 scale-free features: returns, volatility, ATR, RSI, MACD, Bollinger position, volume z-score, gap, engulfing/hammer/doji, 20-bar breakouts, causal double bottom/top |
| Labels | `features/labels.py` | Triple-barrier labels matching the real bracket orders (TP 2 ATR, SL 1 ATR, max 8 bars, never across the close) |
| Model | `models/gbm.py` | LightGBM with purged + embargoed expanding walk-forward validation; production model refit on all data |
| Regime | `models/regime.py` | Gaussian mixture on volatility/trend; highest-volatility state switches entries off |
| News | `models/sentiment.py` | FinBERT (if installed) or a finance lexicon; ambiguous high-impact headlines optionally escalated to Claude; per-symbol exponentially decayed score |
| Fusion | `strategy/ensemble.py` | Probability threshold + expected-value check + regime gate + news veto / boost / exit |
| Risk | `risk/officer.py`, `risk/sizing.py` | Daily drawdown halt, kill switch, latency and stale-data guards, exposure caps, session windows; volatility-targeted, quarter-Kelly-capped sizing |
| Execution | `execution/broker.py` | Alpaca bracket orders with idempotent client order IDs; a simulator with slippage, commissions, gap fills and stop-first ambiguity handling |
| Backtest | `backtest/engine.py`, `backtest/metrics.py` | Event-driven replay through the real engine; Sharpe, Sortino, drawdown, profit factor, equal-weight buy-and-hold benchmark |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"          # add ,nlp for FinBERT and ,llm for Claude escalation
```

### 1. Try the full pipeline offline (no API keys)

```bash
sentinel fetch --synthetic --days 300      # synthetic prices + headlines
sentinel train --holdout-days 10           # walk-forward report + production model
sentinel backtest                          # out-of-sample backtest -> reports/backtest_*/
sentinel replay --days 10                  # live code path on the held-out days
streamlit run dashboard/app.py -- --db data/replay.db
```

Synthetic results tell you the plumbing works. They say nothing about real markets.

### 2. Real data and paper trading

1. Create a free Alpaca account, switch to **Paper Trading**, and generate API keys.
2. `cp .env.example .env`, then fill in `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
3. Pick your symbols and timeframe in `config/config.yaml`.

```bash
sentinel fetch                             # ~2 years of 15-min bars + news (takes a few minutes)
sentinel train --holdout-days 20
sentinel backtest
sentinel replay --days 20                  # does the live path agree with the backtest?
sentinel train                             # refit on everything before going live
sentinel live                              # paper trading; Ctrl+C to stop
streamlit run dashboard/app.py             # in a second terminal
```

Or run the bot and the dashboard together with `docker compose up -d` (dashboard on http://localhost:8501).

**Retraining:** markets drift, so schedule a weekly refit outside market hours, e.g. cron `0 2 * * 6 cd /path && sentinel fetch && sentinel train`, then restart the bot.

### Other commands

```bash
sentinel status                            # account, positions, market clock, recent risk events
sentinel flatten                           # cancel orders + close positions in the configured symbols
touch KILL                                 # kill switch: bot flattens and exits at its next tick
```

## How a trade happens

1. A 15-minute bar completes. Features are computed on a rolling 300-bar window. A test checks that they match the training features exactly.
2. The model outputs *P(take-profit is hit before the stop)*. An entry needs `prob ≥ entry_threshold` and a positive expected value in R units: `p·(TP/SL) − (1−p) ≥ min_edge`.
3. The regime filter blocks entries in the most volatile state. A decayed news score ≤ −0.5 vetoes longs; ≥ +0.5 scales the size up by 1.25×.
4. Sizing: `risk$ = equity × min(0.5%, ¼·Kelly)`, `shares = risk$ / (1 ATR stop)`, capped at 20% notional and the remaining gross-exposure headroom.
5. The risk officer checks halts, latency, session timing, open positions and exposure. Then a **bracket order** (market entry + TP limit + SL stop) is sent with the client ID `snt-<SYMBOL>-<bar time>`, so a reconnect can never double-enter.
6. Exits: TP, SL, a strongly negative headline (checked on arrival in live mode), the 15:50 flatten, or a drawdown or kill-switch halt.

## Validation choices

These are deliberate departures from a "throw every model at it" design:

- **Purged, embargoed walk-forward.** Any training row whose label window could overlap the test fold is dropped, plus an 8-bar embargo. The backtest only trades on predictions from models that never saw that period. (Full combinatorial purged CV is a natural next step. Walk-forward is the closer match to how the bot is actually retrained.)
- **A Gaussian mixture, not an HMM, for regimes.** HMM state decoding (Viterbi or smoothing) uses the whole sequence, which quietly leaks future volatility into a backtest. A GMM classifies each bar from its own features.
- **Labels that mirror the orders.** The model is trained on exactly the event the bracket order pays out on, with the same ATR distances and the same session cut-off. A bar that touches both barriers counts as a loss, and the simulator makes the same assumption.
- **One engine for every mode.** The backtester replays events through the production `TradingEngine` rather than a separate vectorized framework, so no logic is reimplemented.
- **A train/serve parity test.** `test_train_serve_parity` feeds bars one at a time through the live signal path and asserts the probabilities equal the batch pipeline's to 1e-6.
- **Benchmarked.** Every report includes equal-weight buy-and-hold of the same symbols over the same window.

Deep sequence models (TFT/LSTM) and 1D-CNN or DTW pattern matchers are left out on purpose. On a few years of 15-minute bars they usually overfit compared with gradient boosting on engineered features. The `ModelBundle` interface makes them easy to add and A/B against the baseline later.

## Safety

- **Paper by default.** Real money needs all three: `mode: live` in the config, `SENTINEL_LIVE_CONFIRM=yes` in the environment, *and* `sentinel live --confirm-live`. Missing any one, the bot refuses to start.
- **Use a dedicated account.** The bot only manages the symbols in its config, but daily drawdown is measured on total account equity.
- **Hard limits:** −2% daily drawdown flattens and halts until the next session. The `KILL` file flattens and stops. Slow broker calls or stale bars block entries. Everything is flattened before the close, so no overnight risk.
- **Day-trading rules:** FINRA's pattern-day-trader rule ($25k minimum) was eliminated effective June 4, 2026, but brokers have until October 2027 to migrate. If your account still enforces the old limit, set `risk.pdt_guard: true`.
- **Data feed:** the free IEX feed carries only part of US volume. Train and trade on the same feed (`data.feed`). Volume features from IEX are not comparable to consolidated SIP volume.
- Paper trade for at least 4–8 weeks and compare live fills against `sentinel replay` over the same days before risking capital. Backtests are optimistic: they cannot model queue position, partial fills, or market impact.

This is research and engineering software, not financial advice. Trading can lose more than you expect, and past or simulated performance does not predict future results.

## Tests

```bash
pytest -q          # 36 tests, ~3 s, no network
ruff check src tests dashboard
```

The suite covers: no-lookahead features, triple-barrier edge cases (stop-first, session cut-off), walk-forward purging, news only used after publication, sizing math, every risk rule, simulator fills (gaps, slippage, duplicate IDs), the bar aggregator, Alpaca request construction (with a fake client), a full backtest, a drawdown-halt scenario, train/serve parity, and **the live runner end to end with fake websocket streams**, from minute bars through aggregation, signals, bracket orders and the kill-switch shutdown.

## Project layout

```
config/config.yaml        all tunables (no secrets)
src/sentinel/
  cli.py  engine.py  pipeline.py  config.py  events.py  market_time.py
  data/  features/  models/  strategy/  risk/  execution/  backtest/  live/
dashboard/app.py          Streamlit monitor (read-only on the SQLite audit trail)
tests/                    pytest suite
```

## License

MIT
