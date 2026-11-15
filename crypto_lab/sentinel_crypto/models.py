"""Small fixed candidate library with chronological probability calibration."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

FAMILIES = ("logistic", "hist_gradient", "extra_trees")
THRESHOLDS = (0.52, 0.56, 0.60)


def factory(family: str, seed: int):
    if family == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=500, random_state=seed))
    if family == "hist_gradient":
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, learning_rate=0.04,
            l2_regularization=10, min_samples_leaf=60, early_stopping=False, random_state=seed)
    if family == "extra_trees":
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=120,
            max_depth=4, min_samples_leaf=40, max_features=0.7, n_jobs=1, random_state=seed))
    raise ValueError(family)


@dataclass
class Predictor:
    family: str
    estimator: object
    calibrator: object
    columns: list[str]
    train_last: str
    calibration_last: str
    majority: int
    base_rate: float

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        with threadpool_limits(limits=1):
            p = np.clip(self.estimator.predict_proba(x[self.columns])[:, 1], 1e-5, 1 - 1e-5)
            logit = np.log(p / (1 - p)).reshape(-1, 1)
            return self.calibrator.predict_proba(logit)[:, 1]


def fit(family: str, f: pd.DataFrame, y: pd.Series, cfg: dict, boundary: pd.Timestamp) -> Predictor:
    # Labels for t are known only at open t+h+1, before the next test/validation boundary.
    y = y.reindex(f.index)
    latest_allowed = boundary - pd.Timedelta(hours=4 * (cfg["horizon_bars"] + 1))
    mask = ((f.index < latest_allowed) & (f.index >= boundary - pd.Timedelta(days=cfg["train_days"])) & y.notna())
    x = f.loc[mask]
    yy = (y.loc[mask] > 0).astype(int)
    split = int(len(x) * 0.8)
    purge = cfg["horizon_bars"] + 1
    if split < 300 or len(x) - split < 100 or yy.nunique() != 2:
        raise ValueError("Insufficient training/calibration history")
    estimator = factory(family, cfg["seed"])
    with threadpool_limits(limits=1):
        estimator.fit(x.iloc[:split-purge], yy.iloc[:split-purge])
        raw = np.clip(estimator.predict_proba(x.iloc[split:])[:, 1], 1e-5, 1 - 1e-5)
        calibrator = LogisticRegression(C=1.0, random_state=cfg["seed"])
        calibrator.fit(np.log(raw / (1 - raw)).reshape(-1, 1), yy.iloc[split:])
    return Predictor(family, estimator, calibrator, list(x.columns), str(x.index[split-purge-1]),
                     str(x.index[-1]), int(yy.mean() >= 0.5), float(yy.mean()))
