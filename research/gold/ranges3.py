import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from load import daily
g=daily("GC=F"); o,h_,l,c=g["open"],g["high"],g["low"],g["close"]
r=np.log(c).diff()
gvz=daily("^GVZ")["close"].reindex(g.index).ffill()/100/np.sqrt(252)
gk = (0.5*np.log(h_/l)**2 - (2*np.log(2)-1)*np.log(c/o)**2).clip(lower=1e-8) + np.log(o/c.shift())**2   # Garman-Klass + overnight
ew = np.sqrt((r**2).ewm(alpha=0.06).mean())
F = pd.DataFrame({"gk1":np.log(gk),"gk5":np.log(gk.rolling(5).mean()),"gk22":np.log(gk.rolling(22).mean()),
                  "ew":np.log(ew**2),"gvz":np.log(gvz**2),"mon":(g.index.dayofweek==4).astype(float)}, index=g.index)  # Friday close -> Monday next
DEV=("2008-06-01","2018-12-31"); HOLD="2019-01-01"; ROLL=750
def sigma_har(h, cols):
    tgt = np.log((r.shift(-1).rolling(h).apply(lambda x:(x**2).sum(), raw=True)).shift(-(h-1)) + 1e-10) if h>1 else np.log(r.shift(-1)**2+1e-10)
    D = F[cols].assign(t=tgt).dropna()
    dev = D.loc[DEV[0]:DEV[1]]
    m = LinearRegression().fit(dev[cols], dev.t)
    # log of squared return is biased (E[log chi2]); fix the scale on dev so that mean(r^2/sigma^2)=1
    pred = pd.Series(m.predict(F[cols].fillna(F[cols].mean())), index=F.index)
    raw = np.exp(pred/2)
    realized = (np.exp(dev.t)).reindex(raw.index)
    k = np.sqrt((realized/raw**2).loc[DEV[0]:DEV[1]].mean())
    return raw*k/np.sqrt(h)  # per-day sigma for horizon h
CANDS = {"current (EWMA+GVZ blend)": lambda h: (ew+gvz)/2,
         "HAR range-based": lambda h: sigma_har(h, ["gk1","gk5","gk22"]),
         "HAR range+GVZ": lambda h: sigma_har(h, ["gk1","gk5","gk22","gvz"]),
         "HAR range+GVZ+EWMA+weekend": lambda h: sigma_har(h, ["gk1","gk5","gk22","gvz","ew","mon"])}
def evaluate(sig, h, L, start, end):
    z = np.log(c.shift(-h)/c)/(sig*np.sqrt(h)); known=z.shift(h)
    lo=known.rolling(ROLL,min_periods=250).quantile((1-L)/2); hi=known.rolling(ROLL,min_periods=250).quantile(1-(1-L)/2)
    x = np.log(c.shift(-h)/c); w=sig*np.sqrt(h)
    T = pd.concat([x, lo*w, hi*w],axis=1).dropna().loc[start:end].iloc[::h]
    a=1-L; inside=(T.iloc[:,0]>=T.iloc[:,1])&(T.iloc[:,0]<=T.iloc[:,2])
    width=(T.iloc[:,2]-T.iloc[:,1])
    winkler = width + (2/a)*((T.iloc[:,1]-T.iloc[:,0]).clip(lower=0)+(T.iloc[:,0]-T.iloc[:,2]).clip(lower=0))
    return inside.mean(), width.mean(), winkler.mean()
for h in (1,5,21):
    print(f"--- horizon {h} day(s): lower interval score = better (sharper and still calibrated)")
    for name,f in CANDS.items():
        s=f(h); row=[]
        for L in (0.8,0.9):
            cd,wd,sd=evaluate(s,h,L,"2010-06-01","2018-12-31"); ch,wh,sh=evaluate(s,h,L,HOLD,"2026-12-31")
            row.append(f"{int(L*100)}%: DEV score {sd*100:.3f} | HOLD cov {ch:.1%} width {wh*100:.2f}% score {sh*100:.3f}")
        print(f"{name:28s} " + " || ".join(row))
