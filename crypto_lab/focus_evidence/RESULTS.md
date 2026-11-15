# Focused real-data results

Protocol: `focus-gold-oil-crypto-2026-11-15-v2`. Model-validation evaluations: 440; policy-validation evaluations: 1302.

No candidate is qualified for execution or proven best in class. The first frozen crypto report remains unchanged.

| Instrument | Non-overlapping forecasts | Accuracy | Pre-test prior accuracy | Brier skill vs prior | Reference policy net |
|---|---:|---:|---:|---:|---:|
| XAUUSD | 505 | 49.3% | 45.9% | -1.4% | +0.00% |
| WTIUSD | — | Data blocked | — | — | Not evaluated |
| WTI_REFERENCE | 62 | 41.9% | 54.8% | -6.6% | Direction only |
| BTCUSDT | 312 | 49.0% | 49.0% | +0.0% | +0.00% |
| ETHUSDT | 312 | 49.7% | 49.7% | +0.0% | -1.17% |
| SOLUSDT | 312 | 48.4% | 48.4% | +0.0% | +0.00% |

Gold and crypto trading results are simulations with assumed costs. Gold selected cash; crypto BTC and SOL also selected cash. The ETH policy lost money. The WTI CFD source has a 940-day gap; its final window is blocked. WTI_REFERENCE is daily continuous-futures direction research, not USOIL CFD P&L.

The new models did not establish a general accuracy or probability-quality improvement. Some development periods improved, while final retrospective results weakened or selected the prior. New candidates therefore remain challengers; the original paper account is preserved.

Gold reference quotes end on 6 November 2026 and are stale for current signals. No current gold or oil CFD recommendation is generated. Broker-specific data and contract terms are still required.

The last retrospective window starts 14 November 2025. Crypto dates were already inspected in v1, so results are exploratory. Comparative intervals are descriptive and do not correct all selection bias. Current prospective challenger forecasts have no matured outcome at inception.

See `summary.json`, `qualification.json`, `all-model-validation.csv`, `all-policy-validation.csv`, per-instrument OOS forecasts, and `../docs/FOCUS_RESEARCH.md` for the full audit trail.
