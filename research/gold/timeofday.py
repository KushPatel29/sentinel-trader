import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import ttest_1samp
from load import hourly, daily
h = hourly("GC=F"); r = np.log(h["close"]).diff().dropna()
r = r[r.abs() < 0.05]                                   # drop session-gap artefacts
hr = r.index.hour
split = r.index[len(r)//2]
A, B = r[r.index < split], r[r.index >= split]
print(f"hourly data {r.index.min():%Y-%m-%d} -> {r.index.max():%Y-%m-%d}; design half ends {split:%Y-%m-%d}")
mA = A.groupby(A.index.hour).mean(); mB = B.groupby(B.index.hour).mean()
tb = pd.DataFrame({"design_avg_bps": mA*1e4, "test_avg_bps": mB*1e4, "test_n": B.groupby(B.index.hour).size()})
print(tb.round(2).to_string())
print("correlation of hourly pattern, design vs test:", round(np.corrcoef(mA.reindex(range(24)).fillna(0), mB.reindex(range(24)).fillna(0))[0,1],3))
# Strategy chosen on design half: long the 4 best hours, short the 4 worst (each hour held 1h), cost 2bps per trade (CFD spread)
best = mA.nlargest(4).index; worst = mA.nsmallest(4).index
cost = 2/1e4
for name, hrs, sign in (("long best 4 hours", best, 1), ("short worst 4 hours", worst, -1)):
    x = sign*B[np.isin(B.index.hour, hrs)] - cost
    t = ttest_1samp(x, 0)
    print(f"TEST {name} {list(hrs)}: avg per trade {x.mean()*1e4:+.2f} bps after costs, trades {len(x)}, total {np.exp(x.sum())-1:+.1%}, t={t.statistic:.2f}")
    x0 = sign*B[np.isin(B.index.hour, hrs)]
    print(f"     before costs: {x0.mean()*1e4:+.2f} bps/trade, t={ttest_1samp(x0,0).statistic:.2f}")
# Day of week on daily data
g = daily("GC=F"); d = np.log(g["close"]).diff().dropna()
for per,(a,b) in {"2006-2018":("2006","2018"),"2019-2026":("2019","2026")}.items():
    x = d.loc[a:b]; print(per, "avg daily return by weekday (bps):", (x.groupby(x.index.dayofweek).mean()*1e4).round(1).to_dict())
