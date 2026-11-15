"""LightGBM probability model with purged, embargoed walk-forward validation.

The model predicts P(take-profit barrier is hit before the stop) for a long entry at this bar's close.
Walk-forward: expanding training window; every training row whose label window could overlap the
test fold is purged, plus an extra embargo. Out-of-sample probabilities feed the backtest, so the
backtest never sees a prediction made by a model that trained on that period."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from sentinel.config import Config
from sentinel.models.dataset import MODEL_FEATURES
from sentinel.models.regime import RegimeModel

log = logging.getLogger(__name__)


def _fit(train: pd.DataFrame, cfg: Config) -> tuple[lgb.LGBMClassifier, RegimeModel]:
    regime = RegimeModel(cfg.regime.n_states).fit(train)
    X = train.assign(regime=regime.predict(train))[MODEL_FEATURES]
    y = train["label"].astype(int)
    params = {"objective": "binary", "random_state": 42, **cfg.model.lgbm}
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X, y)
    return clf, regime


def _trainable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["entry_window"] & df["label"].notna()]


@dataclass
class ModelBundle:
    gbm: lgb.LGBMClassifier
    regime: RegimeModel
    features: list[str]
    meta: dict = field(default_factory=dict)

    def predict(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probabilities, regime states) for rows of a feature frame."""
        reg = self.regime.predict(feats)
        X = feats.assign(regime=reg)[self.features]
        return self.gbm.predict_proba(X)[:, 1], reg

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


def walk_forward(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (oos predictions for every row in the test folds, per-fold metrics)."""
    ts = np.sort(df["ts"].unique())
    gap = cfg.labels.horizon_bars + cfg.model.embargo_bars
    start = cfg.model.min_train_bars
    if start + 10 >= len(ts):
        raise ValueError(f"Not enough history: {len(ts)} bar timestamps, need > {start + 10}. "
                         "Fetch more data or lower model.min_train_bars.")
    preds, metrics = [], []
    fold = 0
    while start < len(ts):
        test_ts = ts[start:start + cfg.model.test_bars]
        train_cut = ts[max(0, start - gap)]                    # purge + embargo
        train = _trainable(df[df["ts"] < train_cut])
        test = df[df["ts"].isin(test_ts)]
        clf, regime = _fit(train, cfg)
        reg = regime.predict(test)
        prob = clf.predict_proba(test.assign(regime=reg)[MODEL_FEATURES])[:, 1]
        preds.append(pd.DataFrame({"ts": test["ts"].reset_index(drop=True),      # keep tz (UTC)
                                   "symbol": test["symbol"].to_numpy(), "prob": prob, "regime": reg,
                                   "fold": fold}))
        ev = test.assign(prob=prob)
        ev = ev[ev["entry_window"] & ev["label"].notna()]
        m = {"fold": fold, "train_rows": len(train), "test_rows": len(ev),
             "test_start": pd.Timestamp(test_ts[0]).isoformat(), "test_end": pd.Timestamp(test_ts[-1]).isoformat(),
             "base_rate": float(ev["label"].mean()) if len(ev) else np.nan}
        if len(ev) and ev["label"].nunique() == 2:
            m["auc"] = float(roc_auc_score(ev["label"], ev["prob"]))
            m["log_loss"] = float(log_loss(ev["label"], ev["prob"].clip(1e-6, 1 - 1e-6)))
            sel = ev[ev["prob"] >= cfg.model.entry_threshold]
            m["signals"] = int(len(sel))
            m["precision_at_threshold"] = float(sel["label"].mean()) if len(sel) else np.nan
        metrics.append(m)
        log.info("fold %d: train=%d test=%d auc=%s", fold, len(train), len(ev), m.get("auc"))
        start += cfg.model.test_bars
        fold += 1
    return pd.concat(preds, ignore_index=True), pd.DataFrame(metrics)


def train_final(df: pd.DataFrame, cfg: Config, wf_metrics: pd.DataFrame | None = None) -> ModelBundle:
    """Fit on all available history for live trading."""
    train = _trainable(df)
    clf, regime = _fit(train, cfg)
    importance = sorted(zip(MODEL_FEATURES, clf.booster_.feature_importance("gain")), key=lambda x: -x[1])
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train), "symbols": sorted(df["symbol"].unique().tolist()),
        "data_start": df["ts"].min().isoformat(), "data_end": df["ts"].max().isoformat(),
        "timeframe_minutes": cfg.timeframe_minutes, "labels": cfg.labels.model_dump(),
        "feature_importance": [(k, float(v)) for k, v in importance],
        "walk_forward": wf_metrics.to_dict("records") if wf_metrics is not None else [],
    }
    return ModelBundle(gbm=clf, regime=regime, features=list(MODEL_FEATURES), meta=meta)
