"""Nested research models with a separate time-ordered calibration segment."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def estimator(name, seed):
    if name.startswith("logistic"):
        c = .02 if name == "logistic_strong" else .1
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=c, max_iter=500, random_state=seed))
    if name.startswith("hist"):
        return HistGradientBoostingClassifier(max_iter=130, learning_rate=.035, max_leaf_nodes=7 if name == "hist_small" else 15,
            min_samples_leaf=100, l2_regularization=20, early_stopping=False, random_state=seed)
    if name == "extra_trees":
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=120, max_depth=5,
            min_samples_leaf=60, max_features=.75, n_jobs=1, random_state=seed))
    raise ValueError(name)


@dataclass
class ForecastModel:
    name: str
    columns: list[str]
    model: object
    calibrator: object
    base_rate: float
    threshold: float
    train_last_label_end: str
    calibration_last_label_end: str
    boundary: str

    def predict(self, x):
        if self.model is None:
            return np.full(len(x), self.base_rate)
        with threadpool_limits(limits=1):
            raw = np.clip(self.model.predict_proba(x[self.columns])[:, 1], 1e-5, 1-1e-5)
            logit = np.log(raw/(1-raw)).reshape(-1, 1)
            return self.calibrator.predict_proba(logit)[:, 1]


def fit_model(name, f, y, label_end, columns, cfg, boundary):
    y, label_end = y.reindex(f.index), label_end.reindex(f.index)
    calibration_start = boundary-pd.Timedelta(days=cfg["calibration_days"])
    train_start = boundary-pd.Timedelta(days=cfg["train_days"])
    train_mask = (f.index >= train_start)&(f.index < calibration_start)&(label_end < calibration_start)&y.notna()
    cal_mask = (f.index >= calibration_start)&(f.index < boundary)&(label_end < boundary)&y.notna()
    tx, cx = f.loc[train_mask, columns], f.loc[cal_mask, columns]
    # Availability screening is fitted on training only. All-NaN features can
    # otherwise fail binning in some sklearn/numpy versions.
    columns = [c for c in columns if tx[c].notna().any()]
    tx, cx = tx[columns], cx[columns]
    ty, cy = (y.loc[train_mask] > 0).astype(int), (y.loc[cal_mask] > 0).astype(int)
    if len(ty) < 300 or len(cy) < 70 or ty.nunique() < 2 or cy.nunique() < 2:
        raise ValueError(f"Insufficient chronological samples: {len(ty)} training, {len(cy)} calibration")
    prior = float(pd.concat([ty, cy]).mean())
    if name == "prior":
        return ForecastModel(name, columns, None, None, prior, .5, str(label_end.loc[train_mask].max()),
                             str(label_end.loc[cal_mask].max()), str(boundary))
    model = estimator(name, cfg["seed"])
    with threadpool_limits(limits=1):
        model.fit(tx, ty)
        raw = np.clip(model.predict_proba(cx)[:, 1], 1e-5, 1-1e-5)
        cal = LogisticRegression(C=10., random_state=cfg["seed"])
        cal.fit(np.log(raw/(1-raw)).reshape(-1, 1), cy)
        pc = cal.predict_proba(np.log(raw/(1-raw)).reshape(-1, 1))[:, 1]
    # Decision threshold uses only the calibration segment; no test-set threshold fitting.
    grid = np.linspace(.45, .55, 21)
    scores = [balanced_accuracy_score(cy, pc >= threshold) for threshold in grid]
    chosen = sorted(zip(scores, grid), key=lambda x: (-x[0], abs(x[1]-.5)))[0][1]
    return ForecastModel(name, columns, model, cal, prior, float(chosen), str(label_end.loc[train_mask].max()),
                         str(label_end.loc[cal_mask].max()), str(boundary))


def fit_library(f, y, label_end, groups, cfg, boundary):
    models = {}
    for group in cfg["feature_sets"]:
        for family in cfg["models"]:
            if family == "prior" and group != "price":
                continue
            name = f"{family}:{group}"
            models[name] = fit_model(family, f, y, label_end, groups[group], cfg, boundary)
    return models


def choose_forecaster(models, probabilities, y, start, end):
    out = []
    valid = (y.index >= start)&(y.index < end)&y.notna()
    yy = (y.loc[valid] > 0).astype(int)
    for name, model in models.items():
        p = probabilities[name].reindex(yy.index)
        out.append({"candidate": name, "brier": float(brier_score_loss(yy, p)),
                    "balanced_accuracy": float(balanced_accuracy_score(yy, p >= model.threshold)),
                    "calibration_threshold": model.threshold, "samples": len(yy)})
    # Brier evaluates the probabilities rather than optimizing profitable-looking accuracy.
    best = min(out, key=lambda r: (r["brier"], r["candidate"]))["candidate"]
    return best, pd.DataFrame(out)
