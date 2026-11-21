"""Tune the daily plan: selection on 2008-2018 only, single test on 2019-2026, plus yearly walk-forward re-selection."""
import warnings; warnings.filterwarnings("ignore")
import itertools, numpy as np, pandas as pd
from load import daily
g=daily("GC=F"); o,lo,c=g["open"],g["low"],g["close"]
r=np.log(c).diff(); gvz=daily("^GVZ")["close"].reindex(g.index).ffill()/100/np.sqrt(252)
s=((np.sqrt((r**2).ewm(alpha=0.06).mean()))+gvz)/2
COST=2.5/1e4; SLIP=1/1e4; FIN=0.06/252
ma={n:c.rolling(n).mean() for n in (20,50,100,200)}
TRENDS={"50&200":(c>ma[50])&(c>ma[200]), "200":c>ma[200], "100":c>ma[100], "20&50":(c>ma[20])&(c>ma[50])}
def run(trend, stop_sd, exit_on_trend, confirm):
    up=TRENDS[trend]
    if confirm>1: up=up.rolling(confirm).sum()==confirm
    O,L,C,S,U=o.values,lo.values,c.values,s.values,up.values
    n=len(C); daily_ret=np.zeros(n); trades=[]
    inpos=False; pend=False; entry=best=0; days=0; ent_i=0; stop_dist=0
    for i in range(1,n):
        if pend:
            inpos=True; pend=False; entry=O[i]; best=C[i-1]; days=0; ent_i=i
            stop_dist=stop_sd*S[i-1]
            daily_ret[i]-=COST
            prev=entry
        else:
            prev=C[i-1]
        if inpos:
            stop=best*np.exp(-stop_sd*S[i-1])
            if L[i]<=stop:
                px=min(stop,O[i]); daily_ret[i]+=np.log(px/prev)-SLIP
                trades.append((g.index[ent_i], np.log(px/entry)-COST-SLIP-FIN*days, stop_dist)); inpos=False; continue
            daily_ret[i]+=np.log(C[i]/prev)-FIN; days+=1; best=max(best,C[i])
            if exit_on_trend and not U[i] and i+1<n:
                pend_exit_px=O[i+1]; daily_ret[i+1]+=np.log(pend_exit_px/C[i])
                trades.append((g.index[ent_i], np.log(pend_exit_px/entry)-COST-FIN*days, stop_dist)); inpos=False
                # skip day i+1 processing for entry: allow re-entry decision on i+1 close
            continue
        if U[i] and not np.isnan(S[i]) and not np.isnan(ma[200].values[i]): pend=True
    return pd.Series(daily_ret,index=g.index), pd.DataFrame(trades,columns=["date","ret","risk"])
def metrics(dr, T, a, b, risk=0.01):
    d=dr.loc[a:b]; t=T[(T.date>=a)&(T.date<=b)]
    yrs=len(d)/252
    sharpe=d.mean()/d.std()*np.sqrt(252) if d.std()>0 else 0
    # account return when each trade risks `risk` of equity: position = risk / stop distance (capped at 1x... allow up to 3x like a CFD)
    f=(risk/t.risk).clip(upper=3.0); acct=(1+f*(np.exp(t.ret)-1)).prod()
    eq=np.cumprod(1+f*(np.exp(t.ret)-1)); dd=(eq/np.maximum.accumulate(eq)-1).min() if len(eq) else 0
    return dict(cagr=np.exp(d.sum())**(1/yrs)-1, sharpe=sharpe, trades=len(t), win=(t.ret>0).mean(),
                acct_1pct_cagr=acct**(1/yrs)-1, acct_1pct_maxdd=dd)
grid=list(itertools.product(TRENDS, (1.5,2.0,2.5,3.0,4.0), (True,False), (1,3)))
rows=[]; cache={}
for p in grid:
    dr,T=run(*p); cache[p]=(dr,T)
    dv=metrics(dr,T,"2008-01-01","2018-12-31"); ho=metrics(dr,T,"2019-01-01","2026-12-31")
    rows.append(dict(trend=p[0],stop_sd=p[1],exit_on_trend=p[2],confirm=p[3],dev_sharpe=dv["sharpe"],dev_cagr=dv["cagr"],
                     hold_sharpe=ho["sharpe"],hold_cagr=ho["cagr"],hold_trades=ho["trades"],hold_win=ho["win"],
                     hold_acct1pct=ho["acct_1pct_cagr"],hold_dd1pct=ho["acct_1pct_maxdd"]))
R=pd.DataFrame(rows).sort_values("dev_sharpe",ascending=False); pd.set_option("display.width",250)
print("Top 8 by DEV Sharpe (2008-2018) and what they did on 2019-2026:"); print(R.head(8).round(3).to_string(index=False))
cur=R[(R.trend=="50&200")&(R.stop_sd==2.0)&(R.exit_on_trend)&(R.confirm==1)]
print("\nCurrent live rule:"); print(cur.round(3).to_string(index=False))
print("\nRank correlation dev vs holdout Sharpe across all 80 configs:", round(R.dev_sharpe.corr(R.hold_sharpe,method="spearman"),3))
print("Holdout Sharpe: median", round(R.hold_sharpe.median(),3), "best", round(R.hold_sharpe.max(),3))
# walk-forward re-selection: each January pick best trailing (2008..y-1) Sharpe config, trade year y
wf=[]
for y in range(2012,2027):
    best=max(grid,key=lambda p: (lambda d: d.mean()/d.std() if d.std()>0 else -9)(cache[p][0].loc["2008":str(y-1)]))
    wf.append(cache[best][0].loc[str(y)]); 
W=pd.concat(wf)
for a,b in (("2012","2018"),("2019","2026")):
    x=W.loc[a:b]; print(f"Walk-forward re-selected plan {a}-{b}: CAGR {np.exp(x.sum())**(252/len(x))-1:+.1%} Sharpe {x.mean()/x.std()*np.sqrt(252):.2f}")
bh=r.loc["2019":]; print(f"Buy & hold (no financing) 2019-2026: CAGR {np.exp(bh.sum())**(252/len(bh))-1:+.1%} Sharpe {bh.mean()/bh.std()*np.sqrt(252):.2f}")
R.to_csv("plan_tuning_results.csv",index=False)
