"""Versioned accuracy experiment: rolling refits, adaptive priors and guarded ensembles.

Source v1/v2 reports stay immutable. All candidate scores are retained, including
weak results. No trade orders or P&L are inferred from classification accuracy.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from .config import ROOT, atomic_json, digest
from .focus_data import load_focus, read_focus
from .focus_features import build_features, target
from .focus_models import fit_model
from .focus_research import data_gate, read_context
from .metrics import bootstrap_mean


def load_accuracy(root=ROOT):
    return json.loads((root/"accuracy_protocol.json").read_text())


def adaptive_prior(y, label_end, index, cfg):
    known = pd.DataFrame({"y":(y > 0).astype(float).where(y.notna()),"end":label_end}).dropna()
    observations = pd.Series(known.y.to_numpy(),index=pd.DatetimeIndex(known.end)).sort_index()
    lookback = f"{cfg['adaptive_prior_days']}D"
    n = observations.rolling(lookback).count()
    positive = observations.rolling(lookback).sum()
    posterior = (positive+cfg["prior_pseudocount"]*.5)/(n+cfg["prior_pseudocount"])
    # A target outcome is available at its actual end time, never at its origin.
    # Delay one nanosecond to enforce strict end < prediction timestamp.
    posterior.index += pd.Timedelta(nanoseconds=1)
    return posterior.reindex(index,method="ffill").fillna(.5)


def guarded_ensemble(probabilities, y, label_end, cfg):
    candidates = [c for c in probabilities if c != "guarded"]
    known = y.notna()&label_end.notna()
    losses = pd.DataFrame({c:(probabilities[c]-(y > 0).astype(float))**2 for c in candidates}).loc[known]
    losses.index = pd.DatetimeIndex(label_end.loc[known])
    daily = losses.resample("1D").mean()
    # All losses of a UTC day become available the next day. This is conservative
    # even when an individual target finishes earlier in that day.
    daily.index += pd.Timedelta(days=1)
    mean = daily.rolling(f"{cfg['expert_history_days']}D",min_periods=cfg["expert_min_observed_days"]).mean()
    recent = daily.rolling("60D",min_periods=15).mean()
    gains = mean.rsub(mean["adaptive_prior"],axis=0)
    positive_parts = sum((recent.rsub(recent["adaptive_prior"],axis=0).shift(freq=f"{lag}D") > 0).astype(int) for lag in (0,60,120))
    eligible = (gains >= cfg["expert_min_brier_gain"])&(positive_parts >= 2)
    scores = mean.where(eligible)
    choices = pd.Series("adaptive_prior",index=scores.index,dtype=object)
    valid = scores.notna().any(axis=1)
    choices.loc[valid] = scores.loc[valid].idxmin(axis=1)
    choices = choices.reindex(probabilities.index,method="ffill").fillna("adaptive_prior")
    out = np.array([probabilities.at[t,c] for t,c in choices.items()])
    return pd.Series(out,index=probabilities.index),choices


def score(frame, column, baseline="v2_probability"):
    p = frame[column].clip(1e-6,1-1e-6)
    y = frame.actual_up.astype(int)
    prediction = p >= .5
    base_p = frame[baseline]
    base_pred = base_p >= (frame.v2_threshold if baseline == "v2_probability" else .5)
    correct = (prediction == y).astype(float)
    base_correct = (base_pred == y).astype(float)
    daily = pd.Series((correct-base_correct).to_numpy(),index=pd.to_datetime(frame.timestamp,utc=True).dt.floor("D")).groupby(level=0).mean()
    loss_gain = pd.Series(((base_p-y)**2-(p-y)**2).to_numpy(),index=daily_index(frame)).groupby(level=0).mean()
    confident = (p-.5).abs() >= .03
    return {"samples":len(frame),"accuracy":float(accuracy_score(y,prediction)),
            "balanced_accuracy":float(balanced_accuracy_score(y,prediction)),
            "brier":float(brier_score_loss(y,p)),
            "auc":float(roc_auc_score(y,p)) if y.nunique()>1 else None,
            "baseline_accuracy":float(accuracy_score(y,base_pred)),"baseline_brier":float(brier_score_loss(y,base_p)),
            "accuracy_change_pp":float((correct.mean()-base_correct.mean())*100),
            "accuracy_change_ci":bootstrap_mean(daily),"brier_improvement_ci":bootstrap_mean(loss_gain),
            "confidence_coverage":float(confident.mean()),
            "confidence_accuracy":float(accuracy_score(y[confident],prediction[confident])) if confident.any() else None}


def daily_index(frame):
    return pd.to_datetime(frame.timestamp,utc=True).dt.floor("D").to_numpy()


def evaluate_accuracy(root: Path=ROOT):
    cfg,focus = load_accuracy(root),load_focus(root)
    folder=root/"reports"/cfg["version"]
    if (folder/"summary.json").exists():
        saved=json.loads((folder/"summary.json").read_text())
        if saved["protocol_sha256"]!=digest(cfg):
            raise ValueError("Accuracy protocol changed after sealing")
        return saved
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/"protocol_locked.json",{"protocol":cfg,"sha256":digest(cfg),"locked_utc":datetime.now(timezone.utc).isoformat()})
    context=read_context(focus,root)
    start=pd.Timestamp(cfg["development_start"])-pd.Timedelta(days=cfg["expert_history_days"])
    end=pd.Timestamp(cfg["retrospective_end"])
    v2folder=root/"focus_evidence"
    assets,attempts={},[]
    release=root/"release_models"
    release.mkdir(exist_ok=True)
    manifest={"version":cfg["version"],"protocol_sha256":digest(cfg),"assets":{},"orders_enabled":False}
    for symbol in cfg["assets"]:
        print(f"Rolling accuracy experiment {symbol}",flush=True)
        px=read_focus(symbol,root)
        spec=focus["assets"][symbol]
        f,groups=build_features(px,spec,context)
        columns=groups["price"]+[c for c in f if c.startswith("flow_")]
        y,label_end=target(px,spec)
        adaptive=adaptive_prior(y,label_end,f.index,cfg)
        raw=pd.DataFrame(index=f.index,columns=["static_prior","ensemble"],dtype=float)
        at=start
        records=[]
        train_cfg={**focus,"train_days":cfg["train_days"],"calibration_days":cfg["calibration_days"]}
        while at<end:
            until=min(at+pd.Timedelta(days=cfg["retrain_days"]),end)
            okay,gate=data_gate(px,spec,at-pd.Timedelta(days=cfg["train_days"]),until)
            if not okay:
                records.append({"boundary":str(at),"status":"data_blocked",**gate})
                at=until
                continue
            try:
                models=[fit_model(name,f,y,label_end,columns,train_cfg,at) for name in cfg["models"]]
                mask=(f.index>=at)&(f.index<until)
                raw.loc[mask,"ensemble"]=np.mean([m.predict(f.loc[mask]) for m in models],axis=0)
                raw.loc[mask,"static_prior"]=models[0].base_rate
                records.append({"boundary":str(at),"status":"fit","samples_predicted":int(mask.sum()),
                                "last_label_end":models[0].calibration_last_label_end})
            except ValueError as exc:
                records.append({"boundary":str(at),"status":"fit_blocked","reason":str(exc)})
            at=until
        raw["adaptive_prior"]=adaptive
        raw["static_prior"]=raw.static_prior.fillna(adaptive)
        raw["ensemble"]=raw.ensemble.fillna(adaptive)
        raw["blend_25"]=.25*raw.ensemble+.75*adaptive
        raw["blend_50"]=.5*raw.ensemble+.5*adaptive
        raw["guarded"],choices=guarded_ensemble(raw,y,label_end,cfg)
        old=pd.read_csv(v2folder/f"{symbol}-oos-forecasts.csv")
        old["timestamp"]=pd.to_datetime(old.timestamp,utc=True)
        old=old[old.timestamp>=pd.Timestamp(cfg["development_start"])].copy()
        old["v2_probability"],old["v2_threshold"]=old.probability,old.threshold
        for name in cfg["candidates"]:
            old[name]=raw[name].reindex(pd.DatetimeIndex(old.timestamp)).to_numpy()
        old=old.dropna(subset=cfg["candidates"])
        development=old[old.timestamp<pd.Timestamp(cfg["development_end"])]
        retrospective=old[old.timestamp>=pd.Timestamp(cfg["development_end"])]
        metrics={name:{"development":score(development,name),"retrospective":score(retrospective,name)} for name in cfg["candidates"]}
        # Frozen selection: maximize full-coverage development accuracy, subject
        # to probability loss no worse than the adaptive-prior development baseline.
        base=metrics["adaptive_prior"]["development"]
        eligible=[name for name in cfg["candidates"] if metrics[name]["development"]["brier"]<=base["brier"]+.0005]
        chosen=max(eligible,key=lambda n:(metrics[n]["development"]["accuracy"],-metrics[n]["development"]["brier"],n))
        for name,m in metrics.items():
            for phase,result in m.items():
                attempts.append({"symbol":symbol,"candidate":name,"phase":phase,
                                 **{k:v for k,v in result.items() if not isinstance(v,dict)}})
        old.to_csv(folder/f"{symbol}-aligned-forecasts.csv",index=False)
        pd.DataFrame(records).to_csv(folder/f"{symbol}-refit-audit.csv",index=False)
        choices.value_counts().rename_axis("expert").to_csv(folder/f"{symbol}-guarded-expert-counts.csv")
        chosen_m=metrics[chosen]["retrospective"]
        # Release artifact is a new prospective fit; it cannot change old scores.
        production_models=[fit_model(name,f,y,label_end,columns,train_cfg,end) for name in cfg["models"]]
        last_choice=str(choices.iloc[-1]) if len(choices) else "adaptive_prior"
        bundle={"models":production_models,"profile":chosen,"guarded_expert":last_choice,"config":cfg,"spec":spec,
                "symbol":symbol,"fit_before":str(end),"source_last_close":str(px.index[-1]+pd.Timedelta(spec["interval"])),
                "columns":columns,"orders_enabled":False}
        file=release/f"{symbol}.joblib"
        joblib.dump(bundle,file,compress=3)
        import hashlib
        manifest["assets"][symbol]={"file":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
            "profile":chosen,"fit_before":str(end),"source_last_close":bundle["source_last_close"],
            "execution_ready":False,"retrospective_accuracy":chosen_m["accuracy"],"prior_accuracy":chosen_m["baseline_accuracy"]}
        assets[symbol]={"selected_profile":chosen,"metrics":metrics,"selected_retrospective":chosen_m,
                        "model_fit_events":sum(r["status"]=="fit" for r in records),"blocked_fit_events":sum(r["status"]!="fit" for r in records),
                        "interpretation":cfg["interpretation"],"execution_ready":False}
        print(f"  selected on development: {chosen}; retrospective accuracy {chosen_m['accuracy']:.2%}, v2 {chosen_m['baseline_accuracy']:.2%}, change {chosen_m['accuracy_change_pp']:+.2f} pp",flush=True)
    summary={"version":cfg["version"],"protocol_sha256":digest(cfg),"protocol":cfg,"assets":assets,
             "created_utc":datetime.now(timezone.utc).isoformat(),"status":"RETROSPECTIVE_RESEARCH_NOT_EXECUTION_QUALIFIED",
             "classification_threshold":.5,"test_dates_previously_inspected":True,"live_orders_supported":False,
             "limitations":["All comparisons reuse inspected history and are exploratory; future replication is required.",
                "A probability/accuracy improvement does not establish profitable trades after costs.",
                "Commodity reference data is stale and WTI CFD history is incomplete; CFD live inference remains disabled.",
                "Accuracy is reported at full coverage; any confidence subset includes coverage and sample limits.",
                "Monthly refits use only labels whose actual end precedes the refit boundary."]}
    pd.DataFrame(attempts).to_csv(folder/"all-candidate-scores.csv",index=False)
    atomic_json(folder/"summary.json",summary)
    atomic_json(release/"manifest.json",manifest)
    return summary


def release_probability(bundle, px):
    """Local-market features only; no remote macro download in the public quote path."""
    f,_=build_features(px,bundle["spec"],{})
    y,ends=target(px,bundle["spec"])
    adaptive=float(adaptive_prior(y,ends,f.index,bundle["config"]).iloc[-1])
    mean=float(np.mean([m.predict(f.iloc[[-1]])[0] for m in bundle["models"]]))
    values={"static_prior":bundle["models"][0].base_rate,"adaptive_prior":adaptive,"ensemble":mean,
            "blend_25":.25*mean+.75*adaptive,"blend_50":.5*mean+.5*adaptive}
    profile=bundle["profile"]
    if profile=="guarded":
        profile=bundle["guarded_expert"]
    return float(values.get(profile,adaptive))
