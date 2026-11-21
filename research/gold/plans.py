import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from load import daily
g=daily("GC=F"); o,h,l,c=g["open"],g["high"],g["low"],g["close"]
r=np.log(c).diff(); gvz=daily("^GVZ")["close"].reindex(g.index).ffill()/100/np.sqrt(252)
s=((np.sqrt((r**2).ewm(alpha=0.06).mean()))+gvz)/2              # 1-day sigma known at close t
sma50,sma200=c.rolling(50).mean(),c.rolling(200).mean()
up=(c>sma50)&(c>sma200); dn=(c<sma50)&(c<sma200)
COST=2.5/1e4; SLIP=1/1e4; FIN=0.06/252
def zq(z,L): return z.shift(1).rolling(750,min_periods=250).quantile(L)
zL=np.log(c/l.shift(-1))/s; zH=np.log(h.shift(-1)/c)/s
qL90,qH90=zq(zL,0.9),zq(zH,0.9)

def daytrade(direction_series, rr=2.0, stop_q=None):
    """Enter at next open in the given direction (+1/-1/0); stop beyond the 90% next-day level; target rr x risk; else exit at close.
    If stop and target are both touched the same day, assume the stop hit first."""
    out=[]
    for i in range(len(c)-1):
        d=direction_series.iloc[i]
        if d==0 or np.isnan(s.iloc[i]) or np.isnan(qL90.iloc[i]): out.append(np.nan); continue
        e=o.iloc[i+1]; hi,lo,cl=h.iloc[i+1],l.iloc[i+1],c.iloc[i+1]
        if d>0:
            stop=c.iloc[i]*np.exp(-qL90.iloc[i]*s.iloc[i]); 
            if stop>=e: out.append(np.nan); continue          # opened below stop: skip
            tgt=e+rr*(e-stop)
            if lo<=stop: x=np.log(stop/e)-SLIP
            elif hi>=tgt: x=np.log(tgt/e)
            else: x=np.log(cl/e)
        else:
            stop=c.iloc[i]*np.exp(qH90.iloc[i]*s.iloc[i])
            if stop<=e: out.append(np.nan); continue
            tgt=e-rr*(stop-e)
            if hi>=stop: x=np.log(e/stop)-SLIP
            elif lo<=tgt: x=np.log(e/tgt)
            else: x=np.log(e/cl)
        # express as R multiple too
        risk=abs(np.log(stop/e)); out.append((x-COST, risk))
    res=pd.Series([v[0] if isinstance(v,tuple) else np.nan for v in out]+[np.nan], index=c.index)
    risk=pd.Series([v[1] if isinstance(v,tuple) else np.nan for v in out]+[np.nan], index=c.index)
    return res, risk

def swing(direction_series, k=3.0):
    """Hold while the direction signal stays on; chandelier trailing stop k x daily sigma from the best close; financing on longs."""
    pos=0; best=None; rets=[]; 
    for i in range(1,len(c)):
        ret=np.log(c.iloc[i]/c.iloc[i-1])
        if pos!=0:
            stop = best*np.exp(-k*s.iloc[i-1]) if pos>0 else best*np.exp(k*s.iloc[i-1])
            if (pos>0 and l.iloc[i]<=stop) or (pos<0 and h.iloc[i]>=stop):
                px=min(stop,o.iloc[i]) if pos>0 else max(stop,o.iloc[i])
                rets.append(pos*np.log(px/c.iloc[i-1])-COST/2-SLIP); pos=0; best=None; continue
            rets.append(pos*ret-(FIN if pos>0 else 0)); 
            best=max(best,c.iloc[i]) if pos>0 else min(best,c.iloc[i])
            if direction_series.iloc[i]!=pos: rets[-1]-=COST/2; pos=0; best=None
            continue
        rets.append(0.0)
        d=direction_series.iloc[i]
        if d!=0 and not np.isnan(s.iloc[i]): pos=d; best=c.iloc[i]; rets[-1]-=COST/2
    return pd.Series([0.0]+rets, index=c.index)

def stats(x, a, b, per_trade=True):
    y=x.loc[a:b].dropna(); yrs=(pd.Timestamp(b+"-12-31").year-pd.Timestamp(a+"-01-01").year+1) if False else len(x.loc[a:b])/252
    tr=y[y!=0]; eq=np.exp(y.sum()); 
    curve=np.exp(y.cumsum()); dd=(curve/curve.cummax()-1).min()
    streak=0; worst=0
    for v in tr: streak = streak+1 if v<0 else 0; worst=max(worst,streak)
    return dict(cagr=eq**(1/yrs)-1, trades=len(tr), win=(tr>0).mean(), avg_win=tr[tr>0].mean()*1e4, avg_loss=tr[tr<0].mean()*1e4,
                sharpe=(y.mean()/y.std()*np.sqrt(252)) if y.std()>0 else 0, maxdd=dd, worst_streak=worst, t=(tr.mean()/tr.std()*np.sqrt(len(tr))) if len(tr)>1 else 0)
dir_up=up.astype(int); dir_both=up.astype(int)-dn.astype(int); always=pd.Series(1,index=c.index)
cands={}
for rr in (1.0,2.0):
    cands[f"Day trade LONG in uptrend, target {rr:.0f}R"]=daytrade(dir_up,rr)[0]
    cands[f"Day trade BOTH ways with trend, target {rr:.0f}R"]=daytrade(dir_both,rr)[0]
    cands[f"Day trade LONG every day, target {rr:.0f}R"]=daytrade(always,rr)[0]
for k in (2.0,3.0,4.0):
    cands[f"Swing LONG in uptrend, trailing stop {k:.0f} sd"]=swing(dir_up,k)
    cands[f"Swing BOTH ways with trend, trailing stop {k:.0f} sd"]=swing(dir_both,k)
rows=[]
for n,x in cands.items():
    d_=stats(x,"2008","2018"); h_=stats(x,"2019","2026")
    rows.append(dict(plan=n, dev_cagr=d_["cagr"], dev_sharpe=d_["sharpe"], hold_cagr=h_["cagr"], hold_sharpe=h_["sharpe"], hold_trades=h_["trades"],
                     hold_win=h_["win"], avg_win_bps=h_["avg_win"], avg_loss_bps=h_["avg_loss"], hold_maxdd=h_["maxdd"], worst_losing_streak=h_["worst_streak"], t=h_["t"]))
T=pd.DataFrame(rows).sort_values("dev_sharpe",ascending=False); pd.set_option("display.width",260)
print(T.round(3).to_string(index=False))
print("\nToday's state:", "UPTREND" if up.iloc[-1] else "DOWNTREND" if dn.iloc[-1] else "MIXED", "| close", round(c.iloc[-1],1), "| sigma", round(s.iloc[-1],4))
