# Gold, crude oil and crypto upgrade

The commodity target is **spot gold and USOIL-style CFDs**, as requested. A broker has not been selected. The package therefore separates reference-data research, data readiness, and broker execution. No result is called best in class, no broker costs are presented as verified, and no CFD order client is provided.

## Data and instrument identity

- XAUUSD and WTIUSD: [HistData's reference minute bars](https://www.histdata.com/download-free-forex-data/), resampled to hourly bars. Its [data specification](https://www.histdata.com/f-a-q/) states that M1 bars are bid prices, timestamps use fixed EST without daylight saving, and aggregated volume is unavailable. The parser adds exactly five hours to obtain UTC. ZIP integrity is checked and each original file receives a local SHA-256 record. A local hash is not a provider-issued checksum.
- **The WTI feed has a 940-day discontinuity**, and entire recent training windows fail coverage requirements. Those windows are blocked. Data is never silently filled from futures, ETFs or a different broker.
- WTI_REFERENCE: a separate `CL=F` daily direction study starting in 2021. Continuous-futures contract switches can create price changes that are not achievable cash returns. It therefore has **no trading-P&L result** and does not validate USOIL CFDs. The 2020 negative-price period is outside this reference study.
- BTCUSDT, ETHUSDT and SOLUSDT: verified Binance four-hour spot history from the first experiment. Actual buyer-initiated volume can be used here; no analogous volume is invented for gold or oil.
- Daily UUP, TLT, SPY, VIX, GLD and USO are context features, delayed until the following UTC midnight. They represent dollar, bond, broad-equity, volatility, gold-fund and oil-fund price behavior, not complete macroeconomic causation. [GLD](https://www.ssga.com/us/en/individual/etfs/spdr-gold-shares-gld) and [USO](https://www.uscfinvestments.com/uso) are never substituted for the CFD execution price.

The latest HistData commodity file obtained during this run ends on 6 November 2026; it is stale for a current trade. The observer shows **STALE_REFERENCE_NO_SIGNAL** for gold and **DATA_BLOCKED** for WTI instead of displaying a fresh-looking CFD recommendation. The independent daily oil study and crypto data extend later; their timestamps remain visible.

## Modeling and tuning

The candidate library and splits are recorded in `focus_protocol.json`. The model configurations include a prior, two regularized logistic models, two small gradient-boosting models and extra trees, each tested on price-only or expanded context features. It is a bounded search, not an uncontrolled attempt to find a profitable chart.

Each model uses a 730-calendar-day history with a later 120-day calibration segment. Training labels must finish before calibration begins; calibration labels must finish before validation begins. Purging uses the actual target-end timestamp, including session closures, rather than assuming that every bar is contiguous. Feature-availability screening uses training only. Probability calibration and classification-threshold fitting use pre-validation calibration data. See scikit-learn's [threshold-tuning warning](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html) and [calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

The following 180 days select a forecaster using Brier score. A price-only champion and pre-test prior are scored on the same subsequent samples, so the added context must earn its place. For crypto, an ensemble of the original three model families is refitted on the same evaluation dates as an additional model baseline; that is not a rerun of the v1 frozen trading result. Threshold-tuned accuracy, balanced accuracy, AUC, Brier skill, calibration buckets and confidence coverage are all reported.

The selected validation models remain unchanged throughout each later test window. The last retrospective block starts on 14 November 2025. Gold is a new asset study, while crypto overlaps dates already inspected in v1 and is explicitly **exploratory**, not a newly untouched holdout. Repeated research on these dates cannot establish out-of-sample superiority at the researcher level. New future observations are necessary.

Direction targets are the next four hourly trading bars for commodities, constrained to the same UTC session, six four-hour bars for crypto, and three daily observations for the oil reference. Non-overlapping target windows are selected before counting forecasts. Descriptive 14-observed-day block bootstrap intervals compare accuracy and probability loss with the prior and, where available, the original model-family ensemble. These are not a full multiple-testing correction or proof of future skill.

## Trading experiments

Trading selection is separate from prediction selection. Commodity research supports long and short signals; crypto spot stays long or cash. Hysteresis and a minimum holding period reduce unnecessary probability-threshold turnover. Volatility-regime filtering and a target exposure cap are causal.

CFD entries are restricted to 08:00–16:00 UTC and normal positions close by 20:00 UTC. Historical sessions can differ by broker and season; this is a research convention, not the user's eventual broker calendar. Early closures or missing quotes can still produce a carry, which is counted explicitly. Such a result does not establish complete financing treatment.

The reference simulator pays the assumed ask spread on buys against bid-based CFD bars, plus slippage. Position reversals close the existing trade and open a new one with both costs. Broker CSV imports can supply actual bid/ask opens and closes. Strategy validation uses doubled costs, a minimum trade count, drawdown limits, and positive returns in at least two of three validation subperiods. The subsequent test also reports costs multiplied by four. Cash remains a valid selection. Daily-loss and drawdown controls operate at observed prices and are not guaranteed stop prices.

Gold and crude assumptions are configurable research inputs. Financing, contract size, minimum lot, lot step, margin, oil roll dates/cash adjustments, holiday sessions and actual spread/slippage must be established from the selected broker. [CME gold](https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html) and [WTI](https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.contractSpecs.html) are different instruments with their own contract and settlement rules. They do not identify a CFD broker's terms.

## New forward observations

The original v1 paper portfolio and its SQLite ledger are preserved. A separate fixed-model challenger observer records newly observed crypto quotes and subsequent 24-hour direction outcomes. It never changes positions in the v1 account. Its quote-to-quote target approximates the bar-open training target; the timing difference is disclosed. Only actually observed quotes create observations, and restarts do not backfill earlier forecasts. It is a new prospective experiment, not evidence of success at inception.

Current challenger model selection uses trailing validation, then the selected estimator is refit on all information observable before its prospective inception. That is legitimate for a new forward experiment and does not alter historical scores. No automatic self-modification or real-funds promotion occurs.

## Broker data path

Export hourly quotes into the schema in `broker_prices_template.csv`, with an explicit UTC offset in every timestamp. Use:

```powershell
python -m sentinel_crypto.cli import-broker --file YOUR_QUOTES.csv --symbol XAUUSD
python -m sentinel_crypto.cli import-broker --file YOUR_QUOTES.csv --symbol USOIL
```

The import checks timestamps, OHLC bounds and ask ≥ bid, writes an independent broker dataset, and leaves execution readiness false. `broker_profile.example.json` lists the remaining contract/cost fields. It contains no authentication fields. Importing a file does not automatically qualify or replace the existing reference experiment.

We also checked primary [Dukascopy historical-data documentation](https://www.dukascopy.com/api/data/get/historical-data-export) and [FXCM's market-data repository](https://github.com/fxcm/MarketData). Dukascopy's datafeed was unreachable from this machine during this session; the public FXCM sample listing did not provide the required commodity history. Those sources were not silently claimed as downloaded or verified.
