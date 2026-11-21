"""Do chart/candlestick patterns or look-alike (analog) matching predict gold? Select on the first period, confirm on the second."""
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import json, numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from load import daily, hourly, DATA_DIR
from sentinel.gold.patterns import PATTERN_INFO, all_patterns, analog_forecast

def evaluate(df, horizons, dev, hold, label):
    P = all_patterns(df); c = df["close"]; rows = []
    for h in horizons:
        fwd = np.log(c.shift(-h) / c)
        for key, (name, direction, _) in PATTERN_INFO.items():
            for per, (a, b) in (("dev", dev), ("hold", hold)):
                mask = (P[key] > 0) & fwd.notna()
                sel = fwd[mask].loc[a:b]
                allf = fwd.loc[a:b].dropna()
                if len(sel) < 5:
                    rows.append(dict(data=label, pattern=name, key=key, h=h, period=per, n=len(sel))); continue
                hits = int((np.sign(sel) == direction).sum())
                base = float((np.sign(allf) == direction).mean())
                dret = direction * sel
                rows.append(dict(data=label, pattern=name, key=key, h=h, period=per, n=len(sel), hit=hits / len(sel), base=base,
                                 edge=hits / len(sel) - base, avg_bps=dret.mean() * 1e4,
                                 p=binomtest(hits, len(sel), base, alternative="greater").pvalue))
    R = pd.DataFrame(rows)
    D = R[R.period == "dev"].set_index(["key", "h"]); H = R[R.period == "hold"].set_index(["key", "h"])
    cand = D[(D.n >= 20) & (D.edge > 0) & (D.p < 0.05)].index
    print(f"\n=== {label}: {len(D)} pattern x horizon tests; {len(cand)} looked significant in the first period")
    out = []
    for k in cand:
        hrow = H.loc[k] if k in H.index else None
        out.append(dict(pattern=D.loc[k, "pattern"], h=k[1], dev_n=int(D.loc[k, "n"]), dev_hit=D.loc[k, "hit"], dev_base=D.loc[k, "base"],
                        hold_n=int(hrow.n) if hrow is not None else 0, hold_hit=hrow.hit if hrow is not None else np.nan,
                        hold_base=hrow.base if hrow is not None else np.nan, hold_p=hrow.p if hrow is not None else np.nan))
    O = pd.DataFrame(out)
    if len(O):
        # Benjamini-Hochberg on the confirmation p-values
        O = O.sort_values("hold_p"); m = len(O)
        O["bh_threshold"] = [0.05 * (i + 1) / m for i in range(m)]
        O["confirmed"] = O["hold_p"] <= O["bh_threshold"]
        print(O.round(3).to_string(index=False))
    else:
        print("none")
    return R, O

g = daily("GC=F")
R1, O1 = evaluate(g, (1, 5, 10), ("2003-01-01", "2018-12-31"), ("2019-01-01", "2026-12-31"), "daily")
hh = hourly("GC=F").dropna()
mid = hh.index[len(hh) // 2]
R2, O2 = evaluate(hh, (1, 4, 8), (str(hh.index[300]), str(mid)), (str(mid), str(hh.index[-1])), "hourly")

# analog / look-alike matching
res = {}
for label, close, h, dev, hold, mh in (("daily", g["close"], 5, ("2008", "2018"), ("2019", "2026"), 1000),
                                       ("daily-1d", g["close"], 1, ("2008", "2018"), ("2019", "2026"), 1000),
                                       ("hourly", hh["close"], 4, (str(hh.index[2000]), str(mid)), (str(mid), str(hh.index[-1])), 2000)):
    A = analog_forecast(close, horizon=h, min_history=mh)
    fwd = np.log(close.shift(-h) / close).reindex(A.index)
    for per, (a, b) in (("first", dev), ("second", hold)):
        e = A.loc[a:b].join(fwd.rename("fwd")).dropna().iloc[::h]
        y = (e.fwd > 0).astype(int); pred = (e.analog_mean_return > 0).astype(int)
        acc = float((pred == y).mean()); maj = float(max(y.mean(), 1 - y.mean()))
        auc = float(roc_auc_score(y, e.analog_up_share)) if y.nunique() == 2 else 0.5
        pv = binomtest(int((pred == y).sum()), len(e), maj, alternative="greater").pvalue
        print(f"analog {label} h={h} {per}: n={len(e)} accuracy={acc:.3f} vs always-majority {maj:.3f} AUC={auc:.3f} p={pv:.3f}")
        res[f"{label}_{per}"] = dict(n=len(e), acc=acc, majority=maj, auc=auc, p=float(pv))
R1.to_csv(os.path.join(os.path.dirname(__file__), "pattern_tests_daily.csv"), index=False)
R2.to_csv(os.path.join(os.path.dirname(__file__), "pattern_tests_hourly.csv"), index=False)
json.dump({"daily_candidates": O1.to_dict("records"), "hourly_candidates": O2.to_dict("records"), "analogs": res},
          open(os.path.join(os.path.dirname(__file__), "pattern_results.json"), "w"), indent=1, default=str)
