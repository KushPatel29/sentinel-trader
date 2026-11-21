import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import daily
def run(g, spread_bps, slip_bps, gvz_sym="^GVZ", exclude_rolls=False, label=""):
    o,hi,lo,c=g["open"],g["high"],g["low"],g["close"]
    r=np.log(c).diff(); gvz=daily(gvz_sym)["close"].reindex(g.index).ffill()/100/np.sqrt(252)
    s=((np.sqrt((r**2).ewm(alpha=0.06).mean()))+gvz)/2
    zH=np.log(hi.shift(-1)/c)/s; zL=np.log(c/lo.shift(-1))/s
    q=lambda z,L: z.shift(1).rolling(750,min_periods=250).quantile(L)
    out={}
    for L in (0.5,0.8):
        sell_lvl=c*np.exp(q(zH,L)*s); buy_lvl=c*np.exp(-q(zL,L)*s); nb=c.shift(-1); no=o.shift(-1)
        # if the next open already gaps beyond the level, a stop fills at the open, not at the level
        fill_long=np.maximum(sell_lvl, no); fill_short=np.minimum(buy_lvl, no)
        cost=(spread_bps+2*slip_bps)/1e4
        bl=pd.Series(np.where(hi.shift(-1)>=sell_lvl, np.log(nb/fill_long)-cost,0.0),index=c.index)
        bs=pd.Series(np.where(lo.shift(-1)<=buy_lvl, np.log(fill_short/nb)-cost,0.0),index=c.index)
        if exclude_rolls:
            gap=np.abs(np.log(no/c))
            roll=gap>4*s     # abnormal open gaps (contract switches / data jumps)
            bl[roll]=0; bs[roll]=0
        out[L]=(bl,bs)
    for L,(bl,bs) in out.items():
        for name,x in (("long",bl),("both",bl+bs)):
            for per,(a,b) in {"DEV":("2008","2018"),"HOLD":("2019","2026")}.items():
                y=x.loc[a:b].dropna(); eq=np.exp(y.cumsum()); yrs=len(y)/252
                print(f"{label:30s} {int(L*100)}% {name:5s} {per}: CAGR {eq.iloc[-1]**(1/yrs)-1:+6.1%} Sharpe {y.mean()/y.std()*np.sqrt(252):5.2f} t {y.mean()/y.std()*np.sqrt(len(y)):5.2f} trades {(y!=0).sum()}")
run(daily("GC=F"),2,0.5,label="GC=F gap-aware fills")
run(daily("GC=F"),2,0.5,exclude_rolls=True,label="GC=F gap-aware, no roll days")
run(daily("GLD"),1,0.5,label="GLD (ETF, no contract rolls)")
