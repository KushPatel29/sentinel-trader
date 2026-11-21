import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import daily
g = daily("GC=F"); c=g["close"]; r=np.log(c).diff()
gvz = daily("^GVZ")["close"].reindex(g.index).ffill()/100/np.sqrt(252)
ew = {lam: np.sqrt((r**2).ewm(alpha=1-lam).mean()) for lam in (0.90,0.94,0.97)}
def sig(kind):
    if kind.startswith("ewma"): return ew[float(kind[4:])]
    if kind=="gvz": return gvz
    if kind=="blend": return (ew[0.94]+gvz)/2
DEV=("2009-01-01","2018-12-31"); HOLD=("2019-01-01","2026-12-31")
LEVELS=[0.5,0.8,0.9,0.95]
def evaluate(kind, h):
    s = sig(kind)*np.sqrt(h)           # known at close t
    fwd = np.log(c.shift(-h)/c)
    z = (fwd/s).replace([np.inf,-np.inf],np.nan)
    d = z.loc[DEV[0]:DEV[1]].dropna()
    qs = {L: (d.quantile((1-L)/2), d.quantile(1-(1-L)/2)) for L in LEVELS}   # calibrated on DEV only
    ho = z.loc[HOLD[0]:HOLD[1]].dropna().iloc[::h]
    cov = {L: float(((ho>=qs[L][0])&(ho<=qs[L][1])).mean()) for L in LEVELS}
    # proper score: Gaussian log-likelihood of holdout returns (lower NLL better) on dev for choosing kind
    dd = pd.concat([fwd, s],axis=1).dropna().loc[DEV[0]:DEV[1]].iloc[::h]
    nll = float(np.mean(np.log(dd.iloc[:,1]) + 0.5*(dd.iloc[:,0]/dd.iloc[:,1])**2))
    return nll, cov, qs, len(ho)
for h in (1,5,21):
    rows=[]
    for kind in ("ewma0.9","ewma0.94","ewma0.97","gvz","blend"):
        nll,cov,qs,n=evaluate(kind,h); rows.append((kind,nll,cov,n))
    best=min(rows,key=lambda x:x[1])
    for k,nll,cov,n in rows:
        print(f"h={h:2d} {k:9s} devNLL={nll:+.4f} {'<-- chosen on dev' if k==best[0] else '':18s} HOLD n={n} coverage " + " ".join(f"{int(L*100)}%:{cov[L]:.1%}" for L in LEVELS))
# Daily high/low reach: will tomorrow's high exceed close+q*sigma? calibrate quantiles on dev
s=sig("blend"); hi=np.log(g["high"].shift(-1)/c)/s; lo=np.log(c/g["low"].shift(-1))/s
for name,z in (("next-day HIGH above close",hi),("next-day LOW below close",lo)):
    d=z.loc[DEV[0]:DEV[1]].dropna(); ho=z.loc[HOLD[0]:HOLD[1]].dropna()
    for L in (0.5,0.8,0.9):
        q=d.quantile(L); print(f"{name}: {int(L*100)}% level = {q:.2f} sd; holdout share within = {(ho<=q).mean():.1%}")
