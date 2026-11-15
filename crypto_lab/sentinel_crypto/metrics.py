from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score, log_loss


def performance(path: pd.DataFrame, initial: float = 1.0, trades: int = 0) -> dict:
    r = path["return"].fillna(0)
    daily = (1 + r).resample("1D").prod() - 1
    eq = np.r_[initial, path.equity.to_numpy()]
    dd = eq / np.maximum.accumulate(eq) - 1
    years = len(r) / (365 * 6)
    sd = float(daily.std())
    total = float(eq[-1] / initial - 1)
    return {"total_return": total, "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else -1.0,
            "sharpe": float(daily.mean() / sd * np.sqrt(365)) if sd > 1e-12 else 0.0,
            "max_drawdown": float(dd.min()), "annual_volatility": float(sd * np.sqrt(365)) if np.isfinite(sd) else 0.0,
            "mean_daily_return": float(daily.mean()), "median_daily_return": float(daily.median()),
            "round_trips": int(trades), "bars": len(r), "days": len(daily),
            "mean_exposure": float(path.exposure.mean()) if "exposure" in path else None}


def prediction_metrics(y: pd.Series, prob: pd.Series, horizon: int, majority: int, base_rate: float) -> dict:
    d = pd.concat([y.rename("fwd"), prob.rename("p")], axis=1).dropna().iloc[::horizon]
    if len(d) < 20:
        return {"nonoverlapping_samples": len(d)}
    yy, pp = (d.fwd > 0).astype(int), d.p.clip(1e-6, 1 - 1e-6)
    return {"nonoverlapping_samples": len(d), "accuracy": float(accuracy_score(yy, pp >= 0.5)),
        "balanced_accuracy": float(balanced_accuracy_score(yy, pp >= 0.5)),
        "prior_majority_accuracy": float((yy == majority).mean()),
        "auc": float(roc_auc_score(yy, pp)) if yy.nunique() == 2 else None,
        "brier": float(brier_score_loss(yy, pp)),
        "prior_base_rate_brier": float(brier_score_loss(yy, np.full(len(yy), base_rate))),
        "log_loss": float(log_loss(yy, pp, labels=[0, 1])),
        "mean_predicted": float(pp.mean()), "observed_up_rate": float(yy.mean())}


def bootstrap_mean(daily: pd.Series, *, seed=29, block=14, trials=2000) -> dict:
    """Circular moving-block bootstrap; descriptive, not proof of selection-adjusted alpha."""
    arr = daily.dropna().to_numpy()
    n = len(arr)
    if n < block * 2:
        return {"samples": n, "ci95_daily_mean": None, "prob_mean_positive": None}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(trials, int(np.ceil(n / block))))
    indices = ((starts[..., None] + np.arange(block)) % n).reshape(trials, -1)[:, :n]
    means = arr[indices].mean(axis=1)
    return {"samples": n, "block_days": block, "resamples": trials,
            "ci95_daily_mean": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "prob_mean_positive": float((means > 0).mean()),
            "interpretation": "Descriptive uncertainty; no correction for strategy selection or future regime changes."}


def combine(paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Independent fixed capital sleeves; no imaginary frictionless rebalancing between coins.
    eq = pd.concat({k: p.equity for k, p in paths.items()}, axis=1)
    if eq.isna().any().any():
        raise ValueError("Portfolio constituents are not time aligned")
    out = pd.DataFrame({"equity": eq.mean(axis=1)})
    out["return"] = out.equity.pct_change(fill_method=None)
    out.iloc[0, out.columns.get_loc("return")] = out.equity.iloc[0] - 1
    exposure_numerator = sum(paths[k].exposure * paths[k].equity for k in paths)
    out["exposure"] = exposure_numerator / eq.sum(axis=1)
    return out
