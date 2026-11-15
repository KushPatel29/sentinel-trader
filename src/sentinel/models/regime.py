"""Market-regime detection with a Gaussian mixture on volatility/trend features.

A GMM classifies each bar from its own features, so there is no lookahead. (An HMM's Viterbi /
smoothed states use the whole sequence, which leaks future information into a backtest.)
Components are relabelled by volatility so state 0 = calmest and state n-1 = most volatile."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class RegimeModel:
    def __init__(self, n_states: int = 3, seed: int = 0):
        self.n_states = n_states
        self.gmm = GaussianMixture(n_components=n_states, covariance_type="full", random_state=seed, n_init=3)
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.order: np.ndarray | None = None   # component -> sorted state

    @staticmethod
    def _raw(df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            np.log(df["vol_48"].clip(lower=1e-6)),
            df["ret_26"].fillna(0.0),
            np.log(df["vol_ratio"].clip(lower=1e-3)),
        ])

    def fit(self, df: pd.DataFrame) -> "RegimeModel":
        X = self._raw(df)
        X = X[np.isfinite(X).all(axis=1)]
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.gmm.fit((X - self.mu) / self.sd)
        vol_means = self.gmm.means_[:, 0]
        ranks = np.argsort(np.argsort(vol_means))
        self.order = ranks
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._raw(df)
        ok = np.isfinite(X).all(axis=1)
        out = np.full(len(df), -1, dtype=int)
        if ok.any():
            comp = self.gmm.predict((X[ok] - self.mu) / self.sd)
            out[ok] = self.order[comp]
        return out

    @property
    def high_vol_state(self) -> int:
        return self.n_states - 1
