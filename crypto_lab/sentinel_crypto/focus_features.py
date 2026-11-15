"""Instrument-aware, causal features. Daily context becomes usable at next UTC midnight."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_features(px: pd.DataFrame, spec: dict, context: dict[str, pd.DataFrame] | None = None):
    c, o, h, l = px.close, px.open, px.high, px.low
    per_day = {"1h": 24, "4h": 6, "1d": 1}[spec["interval"]]
    r = c.pct_change(fill_method=None)
    f = pd.DataFrame(index=px.index)
    for n in sorted({1, 3, spec["horizon_bars"], per_day, 5*per_day, 20*per_day, 60*per_day}):
        f[f"ret_{n}"] = c.pct_change(n, fill_method=None)
    for days in (5, 20, 60, 200):
        n = days * per_day
        f[f"ma_distance_{days}d"] = c / c.rolling(n, min_periods=min(n, max(20, n//2))).mean() - 1
    for days in (1, 5, 20):
        n = max(2, days * per_day)
        f[f"vol_{days}d"] = r.rolling(n).std() * np.sqrt(spec["annual_bars"])
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean() / c
    f["range_z"] = (c-c.rolling(per_day*5).mean()) / c.rolling(per_day*5).std().replace(0, np.nan)
    f["body"] = (c-o)/c
    f["wick_skew"] = (h+l-2*c)/(h-l).replace(0, np.nan)
    d = c.diff()
    for n in (2, 14):
        up = d.clip(lower=0).rolling(n).mean()
        total = d.abs().rolling(n).mean()
        f[f"rsi_{n}"] = up/total.replace(0, np.nan)
    f["breakout"] = c / h.shift().rolling(per_day*5).max() - 1
    f["breakdown"] = c / l.shift().rolling(per_day*5).min() - 1
    f["drawdown"] = c / c.rolling(per_day*60).max() - 1
    f["vol_regime"] = f.vol_1d/f.vol_20d.replace(0, np.nan)
    f["hour_sin"] = np.sin(2*np.pi*px.index.hour/24)
    f["hour_cos"] = np.cos(2*np.pi*px.index.hour/24)
    f["weekday_sin"] = np.sin(2*np.pi*px.index.dayofweek/7)
    f["weekday_cos"] = np.cos(2*np.pi*px.index.dayofweek/7)
    f["gap_hours"] = px.index.to_series().diff().dt.total_seconds().div(3600).clip(upper=96)
    price_columns = list(f.columns)
    if spec["source"] == "binance":
        v = px.volume
        f["flow_volume_ratio"] = v / v.rolling(per_day*5).mean().replace(0, np.nan)
        f["flow_imbalance"] = (2*px.taker_base-v).rolling(per_day).sum()/v.rolling(per_day).sum().replace(0, np.nan)
        f["flow_buy_share"] = px.taker_base/v.replace(0, np.nan)
    for key, frame in (context or {}).items():
        daily = pd.DataFrame(index=frame.index)
        for n in (1, 5, 20):
            daily[f"context_{key}_ret{n}"] = frame.close.pct_change(n, fill_method=None)
        daily[f"context_{key}_trend"] = frame.close/frame.close.rolling(60).mean()-1
        if key == "fear":
            daily[f"context_{key}_level"] = frame.close/100
        # A daily bar's date alone is not a release timestamp. Delay conservatively
        # until 00:00 UTC on the following day, after the relevant US closes.
        daily.index = pd.to_datetime(daily.index, utc=True).normalize()+pd.Timedelta(days=1)
        daily["_available"] = daily.index
        aligned = daily.reindex(px.index, method="ffill")
        age = (px.index.to_series()-aligned.pop("_available")).dt.total_seconds()/86400
        aligned.loc[age > 7] = np.nan
        f = f.join(aligned)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f, {"price": price_columns, "context": list(f.columns)}


def target(px: pd.DataFrame, spec: dict):
    h = spec["horizon_bars"]
    times = pd.Series(px.index, index=px.index)
    entry, end = times.shift(-1), times.shift(-(h+1))
    y = px.open.shift(-(h+1))/px.open.shift(-1)-1
    if spec["intraday_only"]:
        same_day = entry.dt.normalize() == end.dt.normalize()
        contiguous = end-entry == pd.Timedelta(hours=h)
        session = (entry.dt.hour >= spec["decision_start_hour"]+1) & (end.dt.hour <= spec["flatten_hour"])
        y = y.where(same_day & contiguous & session)
    return y, end


def exposure_scale(f: pd.DataFrame, cfg: dict):
    size = (cfg["target_annual_vol"]/f.vol_20d.clip(lower=.05)).clip(upper=cfg["max_exposure"])
    return (np.floor(size*10)/10).fillna(0)


def signal_series(prob: pd.Series, threshold: float, direction: str, horizon: int):
    """Causal hysteresis and a minimum holding window reduce probability-boundary churn."""
    output, side, held = [], 0., 0
    for p in prob.to_numpy():
        if not np.isfinite(p):
            side, held = 0., 0
        elif side == 0:
            side = 1. if p >= threshold else -1. if direction == "long_short" and p <= 1-threshold else 0.
            held = 0
        elif held >= horizon and ((side > 0 and p < .5) or (side < 0 and p > .5)):
            side, held = 0., 0
        else:
            held += 1
        output.append(side)
    return pd.Series(output, index=prob.index)


def rule_library(px, f, spec, cfg):
    scale = exposure_scale(f, cfg)
    trend = f.ma_distance_20d
    slow = f.ma_distance_60d
    p = pd.DataFrame(index=px.index)
    p["cash"] = 0.
    p["trend_long"] = (trend > 0).astype(float)
    p["trend_slow_long"] = ((trend > 0)&(slow > 0)).astype(float)
    p["reversion_long"] = ((f.rsi_2 < .15)&(slow > 0)).astype(float)
    p["breakout_long"] = (f.breakout > 0).astype(float)
    if spec["source"] == "histdata":
        p["trend_long_short"] = np.sign(trend).fillna(0)
        p["reversion_long_short"] = np.where(f.range_z < -1.5, 1., np.where(f.range_z > 1.5, -1., 0.))
    p = p.mul(scale, axis=0)
    p.loc[f.vol_regime > 2.5] = 0.
    return p
