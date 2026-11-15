"""Walk-forward machine learning on the daily factor library.

Target: does the price rise over the next `horizon` trading days?
Models: LightGBM + regularised logistic regression, averaged (fixed choice, not picked on test data).
Validation: expanding window, refit every `step` days, training labels purged so none overlap the
test window. Every reported number comes from predictions the model made before seeing the outcome."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sentinel.analyzer.factors import ALL_FACTORS, GROUPS, WARMUP, forward_return

LGB_PARAMS = dict(objective="binary", n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1,
                  random_state=7, n_jobs=1)   # one thread: OpenMP oversubscription stalls small cloud containers


@dataclass
class MLResult:
    oos: pd.DataFrame                     # date, prob_gbm, prob_lr, prob, y, fwd
    metrics: dict
    current_prob: float
    current_date: pd.Timestamp
    group_importance: dict[str, float]
    top_features: list[tuple[str, float]]
    features: list[str] = field(default_factory=list)


def _models(X: pd.DataFrame, y: pd.Series):
    gbm = lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y)
    med = X.median()
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500)).fit(X.fillna(med), y)
    return gbm, lr, med


def _predict(models, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    gbm, lr, med = models
    return gbm.predict_proba(X)[:, 1], lr.predict_proba(X.fillna(med))[:, 1]


def walk_forward(f: pd.DataFrame, horizon: int = 5, min_train: int = 756, step: int = 63) -> MLResult:
    feats = [c for c in ALL_FACTORS if c in f.columns and f[c].notna().mean() > 0.5]
    data = f.iloc[WARMUP:].copy()
    data["fwd"] = forward_return(f["close"], horizon).iloc[WARMUP:]
    data["y"] = (data["fwd"] > 0).astype(float).where(data["fwd"].notna())
    n = len(data)
    if n < min_train + step:
        raise ValueError(f"Need at least {WARMUP + min_train + step} daily bars; this asset has {len(f)}.")

    rows = []
    for start in range(min_train, n, step):
        end = min(start + step, n)
        train = data.iloc[: max(0, start - horizon)]            # purge labels that overlap the test window
        train = train[train["y"].notna()]
        test = data.iloc[start:end]
        models = _models(train[feats], train["y"].astype(int))
        pg, pl = _predict(models, test[feats])
        rows.append(pd.DataFrame({"prob_gbm": pg, "prob_lr": pl, "y": test["y"], "fwd": test["fwd"]},
                                 index=test.index))
    oos = pd.concat(rows)
    oos["prob"] = (oos["prob_gbm"] + oos["prob_lr"]) / 2

    ev = oos[oos["y"].notna()]
    # Non-overlapping sample (every `horizon`-th day) for an honest significance test
    nov = ev.iloc[::horizon]
    hits = int(((nov["prob"] > 0.5) == (nov["y"] == 1)).sum())
    base = float(ev["y"].mean())
    majority = max(base, 1 - base)
    metrics = {
        "horizon_days": horizon,
        "oos_days": int(len(ev)),
        "oos_start": ev.index.min(), "oos_end": ev.index.max(),
        "base_rate_up": base,
        "accuracy": float(((ev["prob"] > 0.5) == (ev["y"] == 1)).mean()),
        "majority_accuracy": majority,
        "auc": float(roc_auc_score(ev["y"], ev["prob"])) if ev["y"].nunique() == 2 else 0.5,
        "auc_gbm": float(roc_auc_score(ev["y"], ev["prob_gbm"])) if ev["y"].nunique() == 2 else 0.5,
        "auc_lr": float(roc_auc_score(ev["y"], ev["prob_lr"])) if ev["y"].nunique() == 2 else 0.5,
        "p_value": float(binomtest(hits, len(nov), majority, alternative="greater").pvalue) if len(nov) else 1.0,
        "independent_samples": int(len(nov)),
    }
    conf = ev[(ev["prob"] - 0.5).abs() >= 0.05]
    metrics["confident_share"] = float(len(conf) / len(ev)) if len(ev) else 0.0
    metrics["confident_accuracy"] = float(((conf["prob"] > 0.5) == (conf["y"] == 1)).mean()) if len(conf) else np.nan
    bins = pd.cut(ev["prob"], [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0])
    metrics["calibration"] = (ev.groupby(bins, observed=True)
                              .agg(predicted=("prob", "mean"), actual_up=("y", "mean"), days=("y", "size"))
                              .reset_index().assign(bucket=lambda d: d["prob"].astype(str)).drop(columns="prob"))

    # Final model on everything labelled -> today's probability
    full = data[data["y"].notna()]
    models = _models(full[feats], full["y"].astype(int))
    pg, pl = _predict(models, data[feats].iloc[[-1]])
    imp = pd.Series(models[0].booster_.feature_importance("gain"), index=feats)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    group_imp = {g: float(imp[[c for c in cols if c in imp.index]].sum()) for g, cols in GROUPS.items()}
    return MLResult(oos=oos, metrics=metrics, current_prob=float((pg[0] + pl[0]) / 2), current_date=data.index[-1],
                    group_importance=group_imp, top_features=list(imp.sort_values(ascending=False).head(10).items()),
                    features=feats)
