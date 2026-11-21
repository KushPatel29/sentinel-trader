import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import daily
g = daily("GC=F"); c=g["close"]; r=np.log(c).diff()
gvz = daily("^GVZ")["close"].reindex(g.index).ffill()/100/np.sqrt(252)
s = (np.sqrt((r**2).ewm(alpha=0.06).mean()) + gvz)/2
hi=(np.log(g["high"].shift(-1)/c)/s); lo=(np.log(c/g["low"].shift(-1))/s)
for name,z in (("HIGH",hi),("LOW",lo)):
    for L in (0.5,0.8,0.9):
        # causal rolling quantile: uses only outcomes already known at t (shift 1 so day t+1 outcome not used)
        q = z.shift(1).rolling(750, min_periods=250).quantile(L)
        ho = pd.concat([z,q],axis=1).dropna().loc["2019":]
        print(f"next-day {name} {int(L*100)}% level: holdout within = {(ho.iloc[:,0]<=ho.iloc[:,1]).mean():.1%}  (current level {q.iloc[-1]:.2f} sd)")
# Strategies on gold, holdout 2019-2026, next-day execution, 2bps per turnover
ret = c.pct_change()
def stats(p, name):
    p=p.loc["2019":]; R = (p.shift(1)*ret).loc["2019":] - p.diff().abs().loc["2019":].fillna(0)*0.0002
    eq=(1+R).cumprod(); yrs=len(R)/252
    print(f"{name:26s} CAGR={eq.iloc[-1]**(1/yrs)-1:6.1%} Sharpe={R.mean()/R.std()*np.sqrt(252):5.2f} MaxDD={(eq/eq.cummax()-1).min():6.1%} invested={(p.loc['2019':]>0).mean():.0%}")
vol=r.rolling(21).std()*np.sqrt(252)
stats(pd.Series(1.0,index=c.index),"Buy & hold")
stats((c>c.rolling(200).mean()).astype(float),"200-day trend")
stats((c>c.rolling(50).mean()).astype(float),"50-day trend")
stats((c.pct_change(126)>0).astype(float),"6-month momentum")
stats(((c>c.rolling(200).mean()).astype(float)*(0.15/vol).clip(upper=1)),"Vol-targeted 200-day")
stats((0.15/vol).clip(upper=1.5),"Vol-targeted hold (15%)")
print("Gold now:", round(c.iloc[-1],1), "| vs 50d", f"{c.iloc[-1]/c.rolling(50).mean().iloc[-1]-1:+.1%}", "| vs 200d", f"{c.iloc[-1]/c.rolling(200).mean().iloc[-1]-1:+.1%}", "| from ATH", f"{c.iloc[-1]/c.cummax().iloc[-1]-1:+.1%}", "| 21d vol", f"{vol.iloc[-1]:.0%}", "| daily sd", f"{s.iloc[-1]:.2%}")
