# Gold validation research

Scripts behind the Gold page's evidence. Every model and setting was chosen on **2006–2018** and tested once on **2019–2026**.

| Script | Question |
|---|---|
| `research.py` | Can gold's direction be predicted at 1/5/21/63 days (logistic, LightGBM, trend, momentum)? |
| `tune.py` | 40 tuned configurations with base and expanded drivers (CFTC positioning, credit, Fed policy, INR/CNY/JPY/EUR, Bitcoin, GVZ). Writes `tuning_results.json` |
| `ranges.py`, `ranges2.py` | Calibration of price ranges and of next-day high/low levels |
| `ranges3.py` | HAR / Garman–Klass volatility upgrade vs the live model (interval score) |
| `profit.py` | Strategy tests with spread, slippage and CFD financing |
| `breakout_check.py` | Stress test that exposed the breakout backtest artefact |
| `plans.py` | 12 daily entry/stop/exit plan designs (day trades and trend swings), chosen on 2006–2018 |
| `timeofday.py` | Hour-of-day and weekday up/down patterns, tested out of sample |

`results.json` is the summary the app shows in **What we tested**.

## Data

Put these files in `research/gold/data/` (or set `GOLD_DATA_DIR`):
- `sentinel_gold_data_v2.json`: Yahoo Finance daily bars since 2000 for GC=F and context symbols (format: `{"daily": {symbol: "epoch,open,high,low,close,adjclose,volume\n..."}}`)
- `sentinel_gold_extra.json`: same format for INR=X, CNY=X, JPY=X, EURUSD=X, HYG, LQD, SHY
- `sentinel_cot_gold.json`: CFTC disaggregated futures-only rows for COMEX gold (code 088691) from `publicreporting.cftc.gov/resource/72hh-3qpy.json`
