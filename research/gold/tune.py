import os
import warnings; warnings.filterwarnings("ignore")
import itertools, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import binomtest
from load import daily, daily_any, cot, DATA_DIR
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sentinel.gold.model import features as base_features
from sentinel.gold.data import CONTEXT

g = daily("GC=F"); idx=g.index
def C(sym): d=daily_any(sym)["close"]; return d.reindex(idx.union(d.index)).ffill().reindex(idx)
ctx = pd.DataFrame({k: C(s) for k,s in CONTEXT.items()}, index=idx); ctx["gld_volume"]=daily("GLD")["volume"].reindex(idx)
XA = base_features(g, ctx)
XB = XA.copy()
cc = cot(); net = (cc.m_money_positions_long_all-cc.m_money_positions_short_all)/cc.open_interest_all
cotd = pd.DataFrame({"mm_net": net, "mm_net_z": (net-net.rolling(156,min_periods=52).mean())/net.rolling(156,min_periods=52).std(),
                     "mm_net_chg4": net.diff(4), "prod_net": (cc.prod_merc_positions_long-cc.prod_merc_positions_short)/cc.open_interest_all})
cotd = cotd.reindex(idx.union(cotd.index)).ffill().reindex(idx)
XB = XB.join(cotd)
XB["credit21"] = (C("HYG")/C("LQD")).pct_change(21)
XB["irx21"] = C("^IRX").diff(21); XB["curve"] = C("^FVX")-C("^IRX")
XB["breakeven63"] = (C("TIP")/C("IEF")).pct_change(63); XB["tips5"] = C("TIP").pct_change(5)
for s,n in (("INR=X","inr21"),("CNY=X","cny21"),("JPY=X","jpy21"),("EURUSD=X","eur21")): XB[n]=C(s).pct_change(21)
XB["btc21"] = C("BTC-USD").pct_change(21)
XB["gvz_chg5"] = ctx["gvz"].pct_change(5)
XB = XB.replace([np.inf,-np.inf],np.nan)
SETS = {"base": list(XA.columns), "expanded": list(XB.columns)}
c = g["close"]

def lgbm(params): return lambda: lgb.LGBMClassifier(objective="binary", verbose=-1, n_jobs=1, random_state=1, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, **params)
MODELS = {
  "logit_C0.01": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)),
  "logit_C0.05": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
  "logit_C0.2":  lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=2000)),
  "lgbm_small":  lgbm(dict(n_estimators=150, learning_rate=0.03, num_leaves=5, min_child_samples=100, reg_lambda=10)),
  "lgbm_medium": lgbm(dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=60, reg_lambda=5)),
}
def walk(X, feats, h, mk, years):
    y = (c.shift(-h)/c-1>0).astype(float).where(c.shift(-h).notna())
    D = X[feats].assign(y=y); out=[]
    for yr in years:
        tr = D[D.index < f"{yr}-01-01"].iloc[:-h-1]; tr=tr[tr.y.notna()]
        if len(tr) < 750: continue
        te = D[D.index.year==yr]
        med = tr[feats].median().fillna(0)
        m = mk(); m.fit(tr[feats].fillna(med).fillna(0), tr.y.astype(int))
        out.append(pd.DataFrame({"p": m.predict_proba(te[feats].fillna(med).fillna(0))[:,1], "y": te.y, "base": tr.y.mean()}, index=te.index))
    return pd.concat(out)
def score(E, h):
    e = E.dropna().iloc[::h]
    bs = brier_score_loss(e.y, e.p); bb = brier_score_loss(e.y, e.base)
    hits=int(((e.p>0.5)==(e.y==1)).sum()); maj=max(e.y.mean(),1-e.y.mean())
    return dict(n=len(e), brier_skill=1-bs/bb, auc=roc_auc_score(e.y,e.p), acc=hits/len(e), maj=maj,
                p=binomtest(hits,len(e),maj,alternative="greater").pvalue)
results=[]
import json
DUMP={'dev':[], 'selected':[]}
for h in (1,5,21,63):
    dev=[]
    for (sn,feats),(mn,mk) in itertools.product(SETS.items(), MODELS.items()):
        E = walk(XB, feats, h, mk, range(2008,2019)); s=score(E,h); dev.append((sn,mn,s))
        DUMP['dev'].append(dict(h=h, features=sn, model=mn, **{k:float(v) for k,v in s.items()}))
        print(f"h={h:2d} DEV {sn:8s} {mn:12s} skill={s['brier_skill']:+.4f} auc={s['auc']:.3f} acc={s['acc']:.3f} maj={s['maj']:.3f}", flush=True)
    best = max(dev, key=lambda x: x[2]["brier_skill"])
    sn, mn, _ = best
    E = walk(XB, SETS[sn], h, MODELS[mn], range(2019,2027)); s = score(E,h)
    # confidence filter threshold chosen on dev
    Ed = walk(XB, SETS[sn], h, MODELS[mn], range(2008,2019)).dropna().iloc[::h]
    best_t, best_gain = None, 0
    for t in (0.03,0.05,0.08,0.1):
        sub = Ed[(Ed.p-0.5).abs()>=t]
        if len(sub) < 30: continue
        gain = ((sub.p>0.5)==(sub.y==1)).mean() - max(sub.y.mean(),1-sub.y.mean())
        if gain > best_gain: best_t, best_gain = t, gain
    eh = E.dropna().iloc[::h]
    conf = eh[(eh.p-0.5).abs()>=best_t] if best_t else eh.iloc[:0]
    cacc = ((conf.p>0.5)==(conf.y==1)).mean() if len(conf) else float("nan")
    cmaj = max(conf.y.mean(),1-conf.y.mean()) if len(conf) else float("nan")
    DUMP['selected'].append(dict(h=h, features=sn, model=mn, holdout={k:float(v) for k,v in s.items()}, confident_threshold=best_t, confident_n=int(len(conf)), confident_acc=float(cacc), confident_majority=float(cmaj)))
    json.dump(DUMP, open(os.path.join(DATA_DIR, 'tuning_results.json'),'w'), indent=1, default=str)
    print(f"==> h={h} SELECTED ON DEV: {sn}/{mn} | HOLDOUT skill={s['brier_skill']:+.4f} auc={s['auc']:.3f} acc={s['acc']:.3f} vs always-majority {s['maj']:.3f} p={s['p']:.3f} n={s['n']} | confident(|p-.5|>={best_t}) n={len(conf)} acc={cacc:.3f} vs {cmaj:.3f}", flush=True)
