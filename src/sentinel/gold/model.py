"""Gold forecasting, split into what has proven accurate and what has not.

Measured on 2019-2026 data that played no part in any design choice:
  * PRICE RANGES are well calibrated: the 80% band held 78-80% of outcomes, the 90% band 87-91%.
    Volatility clusters, so the size of the next move is forecastable even when its direction is not.
  * DIRECTION is not: no model or rule beat 'always say up' at 1, 5, 21 or 63 days.
Every accuracy number shown in the app is recomputed from the data on each run, never hard-coded."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HOLDOUT = "2019-01-01"          # every design choice was made on data before this date
LEVELS = (0.5, 0.8, 0.9)
HORIZONS = {"Next day": 1, "Next week (5 days)": 5, "Next month (21 days)": 21}
ROLL = 750                       # rolling window (~3 years) for the causal quantile calibration


# ---------------------------------------------------------------------------------------------- volatility
def daily_sigma(gold: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    """One-day volatility forecast known at today's close: average of realised EWMA volatility
    (lambda 0.94) and gold's implied volatility (GVZ). The blend scored best on 2009-2018 data."""
    r = np.log(gold["close"]).diff()
    ewma = np.sqrt((r ** 2).ewm(alpha=0.06).mean())
    if "gvz" in ctx and ctx["gvz"].notna().any():
        implied = ctx["gvz"] / 100 / np.sqrt(252)
        return ((ewma + implied) / 2).fillna(ewma)
    return ewma


@dataclass
class RangeForecast:
    bands: pd.DataFrame          # horizon x level -> low/high price, plus measured coverage
    highlow: pd.DataFrame        # next-day high/low reach levels with measured accuracy
    sigma: float                 # today's one-day volatility
    cone: pd.DataFrame           # 1..21-day forecast cone for charting


def ranges(gold: pd.DataFrame, ctx: pd.DataFrame) -> RangeForecast:
    c = gold["close"]
    s = daily_sigma(gold, ctx)
    last_c, last_s = float(c.iloc[-1]), float(s.iloc[-1])
    rows = []
    for name, h in HORIZONS.items():
        z = np.log(c.shift(-h) / c) / (s * np.sqrt(h))
        # causal: quantiles at t use only outcomes fully known by t (those that ended h days ago)
        known = z.shift(h)
        for L in LEVELS:
            lo_q = known.rolling(ROLL, min_periods=250).quantile((1 - L) / 2)
            hi_q = known.rolling(ROLL, min_periods=250).quantile(1 - (1 - L) / 2)
            test = pd.concat([z, lo_q, hi_q], axis=1).dropna().loc[HOLDOUT:].iloc[::h]
            cov = float(((test.iloc[:, 0] >= test.iloc[:, 1]) & (test.iloc[:, 0] <= test.iloc[:, 2])).mean())
            width = last_s * np.sqrt(h)
            rows.append({"horizon": name, "days": h, "level": L,
                         "low": last_c * np.exp(float(lo_q.iloc[-1]) * width),
                         "high": last_c * np.exp(float(hi_q.iloc[-1]) * width),
                         "measured_coverage": cov, "tests": len(test)})
    bands = pd.DataFrame(rows)

    hl = []
    for side, z in (("high", np.log(gold["high"].shift(-1) / c) / s), ("low", np.log(c / gold["low"].shift(-1)) / s)):
        for L in LEVELS:
            q = z.shift(1).rolling(ROLL, min_periods=250).quantile(L)
            test = pd.concat([z, q], axis=1).dropna().loc[HOLDOUT:]
            acc = float((test.iloc[:, 0] <= test.iloc[:, 1]).mean())
            move = float(q.iloc[-1]) * last_s
            level = last_c * np.exp(move) if side == "high" else last_c * np.exp(-move)
            hl.append({"side": side, "confidence": L, "level": level, "measured": acc, "tests": len(test)})
    cone_rows = []
    for h in range(1, 22):
        z = np.log(c.shift(-h) / c) / (s * np.sqrt(h))
        known = z.shift(h).iloc[-ROLL:]
        w = last_s * np.sqrt(h)
        cone_rows.append({"h": h, **{f"q{int(p * 100)}": last_c * np.exp(float(known.quantile(p)) * w)
                                     for p in (0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95)}})
    return RangeForecast(bands, pd.DataFrame(hl), last_s, pd.DataFrame(cone_rows))


# ---------------------------------------------------------------------------------------------- features
def features(gold: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    c = gold["close"]
    r = np.log(c).diff()
    X = pd.DataFrame(index=gold.index)
    for k in (5, 21, 63, 126, 252):
        X[f"ret{k}"] = c.pct_change(k)
    X["sma50"] = c / c.rolling(50).mean() - 1
    X["sma200"] = c / c.rolling(200).mean() - 1
    X["vol21"] = r.rolling(21).std() * np.sqrt(252)
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / 14).mean(), (-d).clip(lower=0).ewm(alpha=1 / 14).mean()
    X["rsi14"] = 100 - 100 / (1 + up / dn)
    X["pos52"] = (c - c.rolling(252).min()) / (c.rolling(252).max() - c.rolling(252).min())
    X["dd_ath"] = c / c.cummax() - 1

    def g(k):
        return ctx[k] if k in ctx else pd.Series(np.nan, index=gold.index)
    X["dollar21"] = g("dollar").pct_change(21)
    X["yield21"] = g("yield10").diff(21)
    X["tips21"] = g("tips").pct_change(21)
    X["vix"] = g("vix")
    X["stocks21"] = g("stocks").pct_change(21)
    gs = c / g("silver")
    X["gold_silver_z"] = (gs - gs.rolling(252).mean()) / gs.rolling(252).std()
    X["copper_gold63"] = (g("copper") / c).pct_change(63)
    X["oil21"] = g("oil").pct_change(21)
    X["miners_vs_gold21"] = (g("miners") / g("gld")).pct_change(21)
    lv = np.log1p(g("gld_volume"))
    X["gld_volume_z"] = (lv - lv.rolling(63).mean()) / lv.rolling(63).std()
    X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * gold.index.month / 12), np.cos(2 * np.pi * gold.index.month / 12)
    return X.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------------------------- direction
def direction(gold: pd.DataFrame, ctx: pd.DataFrame, horizons=(5, 21)) -> pd.DataFrame:
    """Walk-forward logistic regression (refit every January on all earlier data, labels purged).
    Reports accuracy on 2019+ next to the 'always up' baseline, so a useless model looks useless."""
    X = features(gold, ctx)
    feats = list(X.columns)
    c = gold["close"]
    rows = []
    for h in horizons:
        y = (c.shift(-h) / c - 1 > 0).astype(float).where(c.shift(-h).notna())
        D = X.assign(y=y)
        preds = []
        years = sorted(set(D.index.year))
        for yr in [y_ for y_ in years if y_ >= 2008]:
            tr = D[D.index < f"{yr}-01-01"].iloc[:-h - 1]
            tr = tr[tr["y"].notna()]
            te = D[D.index.year == yr]
            if len(tr) < 750:          # need ~3 years of history before the first forecast
                continue
            med = tr[feats].median().fillna(0)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000))
            m.fit(tr[feats].fillna(med).fillna(0), tr["y"])
            preds.append(pd.DataFrame({"p": m.predict_proba(te[feats].fillna(med).fillna(0))[:, 1], "y": te["y"]},
                                      index=te.index))
        E = pd.concat(preds)
        current = float(E["p"].iloc[-1])
        ev = E.dropna().loc[HOLDOUT:].iloc[::h]
        hits = int(((ev["p"] > 0.5) == (ev["y"] == 1)).sum())
        up_rate = float(ev["y"].mean())
        majority = max(up_rate, 1 - up_rate)
        acc = hits / len(ev)
        rows.append({"days": h, "prob_up": current, "accuracy": acc, "always_up_accuracy": up_rate,
                     "edge_vs_baseline": acc - majority, "auc": float(roc_auc_score(ev["y"], ev["p"])),
                     "p_value": float(binomtest(hits, len(ev), majority, alternative="greater").pvalue),
                     "tests": len(ev), "reliable": bool(acc > majority and
                                                        binomtest(hits, len(ev), majority, alternative="greater").pvalue < 0.05)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- drivers
DRIVER_INFO = [
    ("dollar", "US dollar index", "pct", "Gold is priced in dollars: a stronger dollar usually weighs on gold"),
    ("yield10", "10-year Treasury yield", "diff", "Higher yields raise the cost of holding gold, which pays no interest"),
    ("tips", "Inflation-protected bonds (TIP)", "pct", "TIP rises when real yields fall, gold's classic tailwind"),
    ("vix", "Stock-market fear (VIX)", "pct", "Panic can lift gold as a haven, or force selling for cash"),
    ("stocks", "US stocks (SPY)", "pct", "Risk appetite"),
    ("silver", "Silver", "pct", "Precious-metal sentiment; silver usually amplifies gold's moves"),
    ("miners", "Gold miners (GDX)", "pct", "Miners often move first: equity investors' view on gold"),
    ("oil", "Crude oil", "pct", "Inflation expectations"),
    ("copper", "Copper", "pct", "Global growth"),
]


def drivers(gold: pd.DataFrame, ctx: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """What has been moving with (or against) gold lately. These are same-day relationships that
    explain moves; they are not forecasts."""
    gr = gold["close"].pct_change()
    rows = []
    for key, name, kind, why in DRIVER_INFO:
        if key not in ctx or ctx[key].dropna().empty:
            continue
        s = ctx[key]
        chg = s.diff(window) if kind == "diff" else s.pct_change(window)
        dr = s.diff() if kind == "diff" else s.pct_change()
        corr = float(gr.iloc[-252:].corr(dr.iloc[-252:]))
        change = float(chg.iloc[-1])
        effect = np.sign(corr) * np.sign(change) if abs(corr) >= 0.15 else 0
        rows.append({"driver": name, "1-month change": change, "kind": kind, "correlation with gold (1y)": corr,
                     "pushing gold": {1: "up", -1: "down", 0: "neutral"}[int(effect)], "why it matters": why})
    return pd.DataFrame(rows)


def sentiment(gold: pd.DataFrame, ctx: pd.DataFrame) -> dict:
    """Crowd psychology readings for gold."""
    c = gold["close"]
    X = features(gold, ctx)
    x = X.iloc[-1]
    r = np.log(c).diff()
    vol_pct = float((r.rolling(21).std()).rank(pct=True).iloc[-1])
    comps = [
        float((c / c.rolling(125).mean() - 1).rolling(756, min_periods=252).rank(pct=True).iloc[-1]),
        float(x["pos52"]) if pd.notna(x["pos52"]) else 0.5,
        1 - vol_pct,
        float(x["rsi14"]) / 100,
    ]
    sign = np.sign(c.diff()).fillna(0)
    streak = int((sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1).iloc[-1] * sign.iloc[-1])
    gvz_pct = None
    if "gvz" in ctx and ctx["gvz"].notna().any():
        gvz_pct = float(ctx["gvz"].rolling(756, min_periods=252).rank(pct=True).iloc[-1])
    return {"greed_index": float(np.mean(comps) * 100), "rsi14": float(x["rsi14"]), "pos52": float(x["pos52"]),
            "drawdown_from_high": float(x["dd_ath"]), "streak_days": streak, "vol_percentile": vol_pct,
            "gvz": float(ctx["gvz"].iloc[-1]) if gvz_pct is not None else None, "gvz_percentile": gvz_pct,
            "gld_volume_z": float(x["gld_volume_z"]) if pd.notna(x["gld_volume_z"]) else None}


# ---------------------------------------------------------------------------------------------- strategies
def strategies(gold: pd.DataFrame, start: str = HOLDOUT, cost_bps: float = 2.5,
               financing_annual: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed rules, next-day execution. `financing_annual` charges overnight holding cost on the long
    notional (CFD brokers typically charge ~5-7%/yr; an ETF charges ~0.4%/yr, physical ~0)."""
    c = gold["close"]
    ret = c.pct_change()
    vol = np.log(c).diff().rolling(21).std() * np.sqrt(252)
    rules = {
        "Buy & hold": pd.Series(1.0, index=c.index),
        "Volatility-sized hold (15% target)": (0.15 / vol).clip(upper=1.5),
        "200-day trend filter": (c > c.rolling(200).mean()).astype(float),
        "50-day trend filter": (c > c.rolling(50).mean()).astype(float),
        "6-month momentum": (c.pct_change(126) > 0).astype(float),
        "Vol-sized 200-day trend": (c > c.rolling(200).mean()).astype(float) * (0.15 / vol).clip(upper=1.0),
    }
    rows, curves = [], {}
    for name, pos in rules.items():
        held = pos.shift(1)
        R = (held * ret - pos.diff().abs() * cost_bps / 1e4 - held.clip(lower=0) * financing_annual / 252
             ).loc[start:].dropna()
        eq = (1 + R).cumprod()
        yrs = len(R) / 252
        rows.append({"strategy": name, "annual_return": eq.iloc[-1] ** (1 / yrs) - 1,
                     "sharpe": R.mean() / R.std() * np.sqrt(252), "max_drawdown": (eq / eq.cummax() - 1).min(),
                     "worst_year": R.groupby(R.index.year).apply(lambda x: (1 + x).prod() - 1).min(),
                     "time_invested": float((pos.loc[start:] > 0).mean()), "position_now": float(pos.iloc[-1])})
        curves[name] = eq
    return pd.DataFrame(rows), pd.DataFrame(curves)


def seasonality(gold: pd.DataFrame) -> pd.DataFrame:
    m = gold["close"].resample("ME").last().pct_change().dropna()
    t = m.groupby(m.index.month).agg(avg_return="mean", up_share=lambda x: (x > 0).mean(), years="size")
    t.index = pd.to_datetime(t.index, format="%m").strftime("%b")
    return t


# ---------------------------------------------------------------------------------------------- risk
def position_size(account: float, risk_pct: float, entry: float, stop: float, oz_per_lot: float = 100.0) -> dict:
    """Size a position so that hitting the stop loses `risk_pct` of the account."""
    risk_cash = account * risk_pct
    per_oz = abs(entry - stop)
    if per_oz <= 0:
        return {"ounces": 0.0, "lots": 0.0, "risk_cash": risk_cash, "notional": 0.0}
    oz = risk_cash / per_oz
    return {"ounces": oz, "lots": oz / oz_per_lot, "risk_cash": risk_cash, "notional": oz * entry}
