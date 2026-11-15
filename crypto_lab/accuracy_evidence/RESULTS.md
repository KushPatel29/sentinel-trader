# Accuracy release v3

This is a retrospective research comparison on dates already inspected in v1/v2. It is not a fresh holdout or a profitable trading qualification.

| Asset | Development-selected profile | Samples | V2 accuracy | V3 accuracy | Change, percentage points | V2 Brier | V3 Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| XAUUSD | guarded | 505 | 49.31% | 51.88% | +2.57 | 0.25540 | 0.24996 |
| BTCUSDT | static_prior | 312 | 49.04% | 49.04% | +0.00 | 0.25146 | 0.25152 |
| ETHUSDT | static_prior | 312 | 49.68% | 49.68% | +0.00 | 0.25028 | 0.25041 |
| SOLUSDT | ensemble | 312 | 48.40% | 48.08% | -0.32 | 0.25020 | 0.25322 |

Gold improves by 2.57 percentage points on these matching retrospective dates, but its descriptive accuracy-difference interval includes zero. Its probability loss also improves; future replication remains necessary. BTC and ETH accuracy are unchanged; SOL is slightly worse. All candidates and uncertainty are published, not just the best row.

Monthly refits use only matured labels. The adaptive prior is indexed by actual label availability. The guarded ensemble uses only loss observations known before the current prediction. The classification threshold is 0.50 for equal-cost direction accuracy; confidence-subset results include coverage.

Models selected on development data remain research artifacts. No profile qualifies broker execution or claims best-in-class performance. The original frozen tests and local paper ledger remain unchanged. Public live data uses Binance’s documented market-data-only endpoint. CFD live inference remains disabled because the reference source is stale/incomplete and broker execution is unverified.
