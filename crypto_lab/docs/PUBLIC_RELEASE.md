# Public research release v3

Target: https://sentinel-trader-kp.streamlit.app/ · Streamlit Community Cloud · `main` · `streamlit_app.py`.

The existing host reported Python 3.14.10 in its deployment log. `requirements.txt` includes the recorded research runtime versions. CI checks the bundled artifacts and root entrypoint on Python 3.14, alongside the existing Python 3.11/3.12 suites.

## What runs publicly

`streamlit_app.py` registers the live research home, commodity audit, archived evidence and preserved asset analyzer. Live quotes use anonymous GET requests to Binance's documented [market-data-only host](https://developers.binance.com/docs/binance-spot-api-docs/faqs/market_data_only). The client allows only server time, book quotes and candle history. No broker account, order endpoint, exchange key or local ledger is involved.

Quotes refresh every 60 seconds while a visitor keeps the page open. History caches for ten minutes and excludes incomplete candles. Research forecasts describe the 24-hour move from the most recent completed four-hour candle close, so part of the target window may already have elapsed at viewing time. They are not a current-price entry recommendation. Artifact hashes are checked before loading. Forecasts stop if candles are stale or models are older than 45 days; source errors are displayed without a synthetic fallback.

Gold/oil reference history does not qualify current CFD inference. The hosted app states these blockers explicitly. Commodity and crypto experiments are versioned separately, and previously inspected dates cannot become a fresh holdout.

## Accuracy method

Run `python -m sentinel_crypto.cli accuracy-research` from `crypto_lab` to reproduce the sealed protocol using the previously downloaded source data. The monthly fit windows purge labels by their actual end timestamps. An adaptive base rate learns only outcomes already available. The guarded ensemble considers only historical candidate losses; candidates are selected on development accuracy with a Brier-loss constraint. Six profiles across four assets are retained in `accuracy_evidence`, including failures and unchanged/worse results.

Models in `release_models` were prospectively fitted before 14 November 2026. These research artifacts do not prove an after-cost trading edge. Refreshing models requires a separately versioned protocol, new causal fits, artifact hashes, tests and a review of genuinely new observations. Do not tune on subsequent outcomes and then describe those outcomes as an untouched test.

## Local workers

The v1 paper account and v2 challenger observer remain separate local processes. Their SQLite files and raw source data are ignored by Git. The hosted site cannot read those ledgers and does not invent a cloud account. Keep the local computer awake and online for continuous collection; Streamlit's web refresh is not an always-on worker. See `crypto_lab/README.md` and `FOCUS_RESEARCH.md` for worker commands.

## Release and rollback

Run the root suite and the crypto suite, then verify the exact PR head's CI before merging into `main`. Streamlit tracks the existing `main` entrypoint. After deployment, verify the public landing page, live quote timestamps, a crypto forecast, the complete accuracy table and the CFD blockers. A source outage should show unavailable while historical evidence still renders. No schema migration or worker-ledger change is part of this release.

If the landing page fails, artifacts cannot load, or a page claims qualified CFD execution, revert the release merge on `main` and let Streamlit redeploy. The prior main revision is `29f8162`. Keep original evidence and local ledgers intact. A manual reboot of the same hosted app can recover a failed build after the underlying issue is corrected.
