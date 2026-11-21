import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import daily
import plans as P            # reuses the tested 'both ways with trend' swing plan (daily returns incl. costs)
x = P.swing(P.dir_both, 2.0).loc["2019":]            # unlevered daily returns of the long+short trend plan
g = daily("GC=F")["close"].loc["2019":]; d = np.log(g).diff().dropna()
print(f"Gold 2019-2026: worst day {d.min():.1%}, best day {d.max():.1%}, worst 3-day move {np.log(g).diff(3).min():.1%}")
print(f"Days the plan made money: {(x>0).mean():.0%} of trading days in a position ({(x!=0).sum()} days); losing days {(x<0).mean():.0%}")
for lev in (1,5,10,20,50):
    eq=1.0; peak=1.0; wiped=None; worst=0
    for dt,ret in x.items():
        eq *= max(0.0, 1 + lev*(np.exp(ret)-1) - (lev-1)*0.06/252*(ret!=0))   # leverage + financing on borrowed part
        peak=max(peak,eq); worst=min(worst, eq/peak-1)
        if eq <= 0.2 and wiped is None: wiped=dt                               # 80% loss ~ margin call / account gone
    print(f"{lev:>2}x leverage: end value {eq:8.2f}x start | worst drawdown {worst:6.1%} | account effectively wiped: {wiped.date() if wiped is not None else 'no'}")
