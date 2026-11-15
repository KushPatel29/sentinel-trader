# LSTM, TFT, trees and a calibrated hybrid

The v4 experiment completed 120 fits: eight model configurations, three chronological boundaries, five usable instruments. These are standard two-layer LSTM, the actual PyTorch Forecasting Temporal Fusion Transformer with a binary classification head, shallow/deeper Random Forest, two LightGBM configurations, Extra Trees and logistic regression. A base-rate forecast and a regularized hybrid are retained in every comparison. WTIUSD CFD history fails its coverage gate and has no fabricated model result.

## Historical research design

The protocol and source SHA-256 hashes are locked in `challenger_evidence/protocol_locked.json`. Development forecasts span 14 May to 14 November 2025. Two later six-month evaluation windows use refits before 14 November 2025 and 14 May 2026, ending 14 November 2026. All of these dates were previously inspected in older experiments, so they are retrospective research rather than a new untouched test.

Every fit separates training, a 45-day early-stop segment and a subsequent 60-day probability-calibration segment. Actual label-end timestamps purge overlapping boundaries. Imputation, scaling and feature screening fit on training data only. Neural sequences contain the last 24 completed observations and end at the decision bar. TFT's decoder receives decision-time calendar fields only; no future market price, target or unknowable covariate is passed to it. Its single direction head is a classification adaptation of TFT, not a claim of multi-horizon quantile trading performance.

The bounded CPU neural configuration uses hidden size 16, at most eight epochs, early stopping, gradient clipping, dropout, weight decay and at most 3,000 uniformly sampled training sequences. This is an actual trained benchmark, not an exhaustive search or a claim that these are optimal neural sizes. All epoch losses and sample counts are retained in fit audits.

The hybrid fits nonnegative, sum-to-one weights on development probabilities with regularization toward equal weights, then shrinks 25% toward the earlier base rate. Its weights remain frozen for later evaluation. The development-selected model minimizes Brier loss; a later-period accuracy winner is not silently substituted. Gold/crypto forecast dates use the previous release's timestamp grid and actual non-overlapping target ends. Matching v3 comparisons report their exact shared sample counts.

## Measured outcome

| Instrument | Selected earlier | Samples shared with v3 | V3 accuracy | V4 selected accuracy |
|---|---|---:|---:|---:|
| BTCUSDT | Hybrid | 311 | 48.87% | 48.87% |
| ETHUSDT | Prior | 311 | 49.52% | 49.52% |
| SOLUSDT | Prior | 311 | 48.23% | 50.80% |
| XAUUSD | Small LightGBM | 505 | 51.88% | 51.49% |

The daily WTI futures-reference study selected a prior forecast and measured 53.33% on 75 observations. It does not establish USOIL CFD execution returns. Individual models sometimes have higher later accuracy—for example, the shallow RF measured 52.09% on ETH and LSTM measured 52.08% on gold—but those rows did not win the frozen development selection. Reporting them as the chosen strategy after seeing later outcomes would overstate the evidence. The complete table, including weak models, is published.

No broad superiority or after-cost profit has been established. The v3 public inference artifacts are preserved. All new models are separate paper challengers.

## New paper collection

New prospective models were fitted with only labels available before their recorded boundary. Artifacts, frozen mixture weights and hashes live in ignored `runtime/challengers`; no credentials exist in this workflow. The worker uses Binance's public market-data-only client. It records actual quotes every minute and has two independent cohorts:

- **Primary bar-close cohort:** new forecasts only within five minutes after a completed four-hour candle, then a non-overlapping 24-hour observation window. Restarting between closes does not backfill a signal.
- **One-time startup diagnostics:** collection begins immediately using the most recent completed candle and a newly observed quote. Its 24-hour quote-origin target differs in timing from bar-open training, and inputs may be up to four hours old. These books never enter the primary accuracy or balances.

Each model/asset/cohort has its own virtual $10,000 book, uses at most 10% exposure, and buys spot only if probability is at least 0.57; otherwise it stays in cash. Paper buys use a fresh observed ask after inference, paper sells use the actual observed bid, and both sides include assumed 10 bps fees plus 5 bps slippage. These are simulated fills and cost assumptions, not verified exchange fills or account fees. Independent virtual books must not be summed into a funded portfolio.

An outcome observed over five minutes late expires for direction scoring. Any hypothetical open position closes at the actual later quote, with the delay disclosed. Missing prices are never invented. Stop requests close simulated positions with fresh quotes and exclude early-cancelled labels from accuracy. A new worker cannot coexist on the same local lease port. Models older than 30 days stop creating forecasts; no self-promotion to real money or silent retraining occurs.

The first real 24-hour labels cannot exist at inception. Pending forecasts, cash decisions and zero closed-trade P&L are not proof of performance. This process requires the computer awake and online. A Streamlit Cloud page cannot see the local ledger or serve as an always-on worker.

## Run and stop

Use a separate Python 3.12 environment so the running web app's packages stay intact:

```powershell
python -m venv .venv-neural
.venv-neural\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv-neural\Scripts\python -m pip install -r requirements-neural.txt
.venv-neural\Scripts\python -m sentinel_crypto.cli challenger-research
.venv-neural\Scripts\python -m sentinel_crypto.cli challenger-prepare
.\start_challengers.ps1 -Python "$PWD\.venv-neural\Scripts\python.exe"
python -m sentinel_crypto.cli challenger-status
python -m sentinel_crypto.cli challenger-stop
```

Do not re-run `challenger-prepare` over an existing inception; it refuses replacement. Preserve the ledger. The installed training environment for this session is outside the repository under the chat's `work/neural-env`. Open the **LSTM · TFT · RF** tab at `http://localhost:8502/` for live local book status. The public site shows packaged historical evidence and correctly reports that local books are unavailable there.

## Primary sources

- [TFT paper by Lim et al.](https://arxiv.org/abs/1912.09363) and [PyTorch Forecasting TFT implementation](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html).
- [PyTorch Forecasting classification loss](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.metrics.point.CrossEntropy.html).
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration) for probability assessment distinct from threshold accuracy.
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) for deterministic, bounded CPU fitting.
- [Binance public market-data-only API](https://developers.binance.com/docs/binance-spot-api-docs/faqs/market_data_only).
