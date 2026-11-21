import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import daily
g=daily("GC=F"); o,hi,lo,c=g["open"],g["high"],g["low"],g["close"]
r=np.log(c).diff(); gvz=daily("^GVZ")["close"].reindex(g.index).ffill()/100/np.sqrt(252)
s=((np.sqrt((r**2).ewm(alpha=0.06).mean()))+gvz)/2
SPREAD=2.0/1e4; SLIP=0.5/1e4; FIN_LONG=0.06/360; FIN_SHORT=0.01/360   # CFD-style: pay ~6%/yr to hold longs overnight
ROLL=750
zH=np.log(hi.shift(-1)/c)/s; zL=np.log(c/lo.shift(-1)/1)/s
def q(z,L): return z.shift(1).rolling(ROLL,min_periods=250).quantile(L)
res={}
# ---- intraday range strategies (enter during next day, exit at next close: no overnight financing)
for L in (0.5,0.8,0.9):
    buy_lvl = c*np.exp(-q(zL,L)*s); sell_lvl=c*np.exp(q(zH,L)*s)
    nb=c.shift(-1)
    fade_long = np.where(lo.shift(-1)<=buy_lvl, np.log(nb/buy_lvl) - SPREAD - SLIP, 0.0)
    fade_short= np.where(hi.shift(-1)>=sell_lvl, np.log(sell_lvl/nb) - SPREAD - SLIP, 0.0)
    brk_long  = np.where(hi.shift(-1)>=sell_lvl, np.log(nb/sell_lvl) - SPREAD - 2*SLIP, 0.0)
    brk_short = np.where(lo.shift(-1)<=buy_lvl, np.log(buy_lvl/nb) - SPREAD - 2*SLIP, 0.0)
    res[f"Fade {int(L*100)}% low (buy dip, exit close)"]=pd.Series(fade_long,index=c.index)
    res[f"Fade {int(L*100)}% both sides"]=pd.Series(fade_long+fade_short,index=c.index)
    res[f"Breakout {int(L*100)}% high (buy, exit close)"]=pd.Series(brk_long,index=c.index)
    res[f"Breakout {int(L*100)}% both sides"]=pd.Series(brk_long+brk_short,index=c.index)
# intraday open->close long every day
res["Intraday long (open->close daily)"]=pd.Series(np.log(c.shift(-1)/o.shift(-1))-SPREAD-SLIP,index=c.index)
# ---- swing strategies held overnight (pay financing)
ret=np.log(c.shift(-1)/c)
vol=r.rolling(21).std()*np.sqrt(252)
def swing(pos, short_ok=False):
    p=pos.fillna(0); turn=p.diff().abs().fillna(p.abs())
    fin=np.where(p>0, p*FIN_LONG, -p*FIN_SHORT)*1.0
    return p*ret - turn*(SPREAD+SLIP) - fin
res["Buy & hold (CFD, financed)"]=swing(pd.Series(1.0,index=c.index))
res["Buy & hold (ETF-style, no financing)"]=pd.Series(1.0,index=c.index)*ret
res["Vol-sized hold 15% (CFD)"]=swing((0.15/vol).clip(upper=1.5))
res["200-day trend (CFD)"]=swing((c>c.rolling(200).mean()).astype(float))
res["200-day trend long/short (CFD)"]=swing(np.sign(c/c.rolling(200).mean()-1))
res["6-month momentum (CFD)"]=swing((c.pct_change(126)>0).astype(float))
d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/2).mean(); dn=(-d).clip(lower=0).ewm(alpha=1/2).mean(); rsi2=100-100/(1+up/dn)
pos=pd.Series(0.0,index=c.index); state=0.0
sma5=c.rolling(5).mean(); sma200=c.rolling(200).mean()
for i in range(len(c)):
    if state==0 and rsi2.iloc[i]<10 and c.iloc[i]>sma200.iloc[i]: state=1.0
    elif state==1 and c.iloc[i]>sma5.iloc[i]: state=0.0
    pos.iloc[i]=state
res["RSI(2) pullback in uptrend (CFD)"]=swing(pos)
tom=((g.index.day>=26)|(g.index.day<=3)).astype(float)
res["Turn of month (CFD)"]=swing(pd.Series(tom,index=c.index))
def stats(x,a,b):
    x=x.loc[a:b].dropna(); act=(x!=0).sum(); yrs=len(x)/252; eq=np.exp(x.cumsum())
    return dict(cagr=eq.iloc[-1]**(1/yrs)-1, sharpe=x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0, mdd=(eq/eq.cummax()-1).min(), trades=int(act), t=x.mean()/x.std()*np.sqrt(len(x)) if x.std()>0 else 0)
rows=[]
for k,x in res.items():
    d_=stats(x,"2008-01-01","2018-12-31"); h_=stats(x,"2019-01-01","2026-12-31")
    rows.append([k,d_["cagr"],d_["sharpe"],d_["trades"],h_["cagr"],h_["sharpe"],h_["mdd"],h_["trades"],h_["t"]])
T=pd.DataFrame(rows,columns=["strategy","DEV CAGR","DEV Sharpe","DEV trades","HOLD CAGR","HOLD Sharpe","HOLD maxDD","HOLD trades","HOLD t"]).sort_values("DEV Sharpe",ascending=False)
pd.set_option("display.width",250); print(T.round(3).to_string(index=False))
