import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
from load import daily

g = daily("GC=F")
idx = g.index
def C(sym): return daily(sym)["close"].reindex(idx.union(daily(sym).index)).ffill().reindex(idx)
c = g["close"]; r = np.log(c).diff()
X = pd.DataFrame(index=idx)
for k in (5,21,63,126,252): X[f"ret{k}"] = c.pct_change(k)
X["mom12_1"] = c.shift(21)/c.shift(252)-1
X["sma50"] = c/c.rolling(50).mean()-1; X["sma200"] = c/c.rolling(200).mean()-1
X["vol21"] = r.rolling(21).std()*np.sqrt(252); X["vol_ratio"] = r.rolling(21).std()/r.rolling(126).std()
d = c.diff(); up=d.clip(lower=0).ewm(alpha=1/14).mean(); dn=(-d).clip(lower=0).ewm(alpha=1/14).mean(); X["rsi14"]=100-100/(1+up/dn)
X["pos52"] = (c-c.rolling(252).min())/(c.rolling(252).max()-c.rolling(252).min()); X["dd_ath"]=c/c.cummax()-1
dxy, tnx, tip, ief, vix, spy, si, hg, cl, gld, gdx, gvz = (C(s) for s in ["DX-Y.NYB","^TNX","TIP","IEF","^VIX","SPY","SI=F","HG=F","CL=F","GLD","GDX","^GVZ"])
X["dxy21"]=dxy.pct_change(21); X["dxy63"]=dxy.pct_change(63)
X["tnx21"]=tnx.diff(21); X["tnx63"]=tnx.diff(63)
X["tip21"]=tip.pct_change(21); X["tip_ief63"]=(tip/ief).pct_change(63)
X["vix"]=vix; X["vix21"]=vix.pct_change(21); X["spy21"]=spy.pct_change(21)
gs=c/si; X["gs_ratio_z"]=(gs-gs.rolling(252).mean())/gs.rolling(252).std()
X["cu_au63"]=(hg/c).pct_change(63); X["oil21"]=cl.pct_change(21)
X["gdx_gld21"]=(gdx/gld).pct_change(21); X["gdx_gld63"]=(gdx/gld).pct_change(63)
gv = daily("GLD")["volume"].reindex(idx); lv=np.log1p(gv); X["gld_vol_z"]=(lv-lv.rolling(63).mean())/lv.rolling(63).std()
X["gvz"]=gvz; X["month_sin"]=np.sin(2*np.pi*idx.month/12); X["month_cos"]=np.cos(2*np.pi*idx.month/12)
X=X.replace([np.inf,-np.inf],np.nan)
FEATS=list(X.columns)
DEV_END, HOLD_START = "2018-12-31", "2019-01-01"
res=[]
def evaluate(h, model):
    y = (c.shift(-h)/c-1 > 0).astype(float).where(c.shift(-h).notna())
    D = X.assign(y=y).loc["2003":]
    preds=[]
    for yr in range(2008, 2027):
        tr = D[D.index < f"{yr}-01-01"]; tr = tr.iloc[:-h-1]; tr=tr[tr.y.notna()]
        te = D[(D.index>=f"{yr}-01-01")&(D.index<f"{yr+1}-01-01")]
        if te.empty: continue
        Xtr = tr[FEATS]
        if model=="lgbm":
            m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.03,num_leaves=7,min_child_samples=60,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,reg_lambda=5,verbose=-1,n_jobs=1,random_state=1).fit(Xtr,tr.y)
            p=m.predict_proba(te[FEATS])[:,1]
        elif model=="logit":
            med=Xtr.median().fillna(0); m=make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=1000)).fit(Xtr.fillna(med).fillna(0),tr.y); p=m.predict_proba(te[FEATS].fillna(med).fillna(0))[:,1]
        elif model=="tsmom":  # sign of 126d return -> fixed prob
            p=np.where(te["ret126"]>0,0.55,0.45)
        elif model=="sma200":
            p=np.where(te["sma200"]>0,0.55,0.45)
        elif model=="base":
            p=np.full(len(te), tr.y.mean())
        preds.append(pd.DataFrame({"p":p,"y":te.y},index=te.index))
    E=pd.concat(preds).dropna()
    out={}
    for name,(a,b) in {"dev":("2008","2018"),"hold":("2019","2026")}.items():
        e=E.loc[a:b].iloc[::h]  # non-overlapping
        acc=((e.p>0.5)==(e.y==1)).mean(); maj=max(e.y.mean(),1-e.y.mean())
        auc=roc_auc_score(e.y,e.p) if e.p.nunique()>1 else 0.5
        pv=binomtest(int(((e.p>0.5)==(e.y==1)).sum()),len(e),maj,alternative="greater").pvalue
        out[name]=(len(e),acc,maj,auc,pv)
    return out
for h in (1,5,21,63):
    for m in ("base","sma200","tsmom","logit","lgbm"):
        o=evaluate(h,m)
        print(f"h={h:2d} {m:6s} | DEV n={o['dev'][0]:4d} acc={o['dev'][1]:.3f} maj={o['dev'][2]:.3f} auc={o['dev'][3]:.3f} p={o['dev'][4]:.3f} | HOLD n={o['hold'][0]:4d} acc={o['hold'][1]:.3f} maj={o['hold'][2]:.3f} auc={o['hold'][3]:.3f} p={o['hold'][4]:.3f}", flush=True)
