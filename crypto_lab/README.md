# Sentinel Crypto Lab

A working crypto research and forward paper-simulation companion to Sentinel Trader. **No profitable edge has been established. This package has no real-order API and needs no exchange keys.**

The first frozen six-month test selected cash for BTC, ETH and SOL: **0.00% net**, while equal-weight full-exposure buy & hold returned **−15.70%**. A volatility-matched baseline returned **−5.17%**. Earlier nested development periods returned **+1.63%** with a **−12.09%** maximum drawdown and 202 round trips. These are simulations after assumed fees and slippage, not earnings. Staying in cash in one falling market is not proof of alpha.

The protocol, all 780 validation evaluations, fold results, prediction metrics and equity curves are in `evidence/`. Historical inputs comprise 38,595 completed four-hour candles across three assets. Monthly archives were verified against published SHA-256 checksums. The future paper track record begins only when the local worker starts; no old trades are backfilled.

## Run locally

Use Python 3.11 or newer. From this directory:

```powershell
python -m pip install -e ".[dev,dashboard]"
python -m sentinel_crypto.cli fetch
python -m sentinel_crypto.cli research
python -m sentinel_crypto.cli prepare-paper
python -m sentinel_crypto.cli paper
```

In a second terminal:

```powershell
python -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8502
```

Or use `start_paper.ps1` and `start_dashboard.ps1`. They start hidden local processes and write PID and log files under `runtime/`. The paper worker is active only while your computer is awake and online. The dashboard is at [localhost:8502](http://localhost:8502).

```powershell
python -m sentinel_crypto.cli status
python -m sentinel_crypto.cli stop
```

The stop command asks the worker to liquidate simulated positions with fresh quotes. If the exchange is unavailable, the ledger retains the position and logs the failure. To resume after a stop, remove `runtime/STOP` and launch `start_paper.ps1`. A single-worker OS lock prevents duplicate portfolios. Restarting preserves the ledger.

## What is implemented

- Ten trading rules, cash and three calibrated ML families at three fixed thresholds: 20 candidates.
- Purged temporal training/calibration, validation-only selection, nested forward development folds and a frozen final 180-day test. Rerunning returns the sealed result; changing its protocol is rejected.
- Causal demand/volume, volatility, trend, candle and crowding-proxy features. News and external sentiment are research topics, not falsely advertised trained inputs.
- Next-open historical simulation, fees/slippage, doubled-cost stress, baselines, drawdown and daily-loss circuit breakers, explicit exposure and no leverage.
- Continuous local forward simulation of all 20 candidates and the selected policy on actual public bid/ask quotes. SQLite transactions, audit logs, restart persistence, stale-signal checks, no historical fills, non-overlapping forward forecasts and a read-only dashboard.
- 27 offline tests covering leakage, timestamps, accounting, risk, restart behavior, duplicate prevention, late signals, outages, transactional rollback, stopping and dashboard rendering.

Current forward policies are selected using all now-past data up to 15 November 2026 04:00 UTC. That includes the historical holdout, which is legitimate **only for this new prospective experiment**. It does not modify or improve the frozen historical score. Fixed policies may lose money; no automatic promotion to real funds exists.

## Research and limits

See [primary-source research](docs/RESEARCH.md) and [architecture](docs/ARCHITECTURE.md). The review covers Freqtrade/FreqAI, CCXT, Hummingbot, NautilusTrader, LEAN, Backtrader, Backtesting.py, vectorbt, Qlib, FinRL, FinBERT, validation literature and data providers. It is a focused review, not exhaustive research on every open-source trading project.

Results are specific to three surviving coins, one venue, a USDT quote currency, a selected interval and assumed costs. Four-hour bars do not model queue position, market impact or partial fills. Prospective quotes can arrive later than a backtest's next open; forward risk is checked more often than historical risk. Block-bootstrap intervals are descriptive and do not correct all selection bias. Cash earns no interest in these tests.

Nothing here guarantees profitability, maximum possible accuracy or suitability for real funds. The next source of evidence is new future data, not repeated tuning of the same holdout.
