"""Causal close-of-candle inputs; every rolling computation ends at the current row."""
from __future__ import annotations
import numpy as np
import pandas as pd

PER_YEAR = 365 * 6


def features(px: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = px.close, px.high, px.low, px.volume
    f = pd.DataFrame(index=px.index)
    r = c.pct_change(fill_method=None)
    for n in (1, 3, 6, 18, 42, 120, 360):
        f[f"ret_{n}"] = c.pct_change(n, fill_method=None)
    for n in (18, 42, 120, 360, 1200):
        f[f"dist_ma_{n}"] = c / c.rolling(n).mean() - 1
    for n in (18, 42, 120):
        f[f"vol_{n}"] = r.rolling(n).std() * np.sqrt(PER_YEAR)
    d = c.diff()
    for n in (2, 14):
        up = d.clip(lower=0).rolling(n).mean()
        down = (-d.clip(upper=0)).rolling(n).mean()
        denom = up + down
        f[f"rsi_{n}"] = (up / denom).where(denom > 0, 0.5)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["candle_body"] = (c - px.open) / c
    f["close_location"] = ((c - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    f["volume_ratio"] = v / v.rolling(42).mean().replace(0, np.nan)
    f["volume_impulse"] = (r * v).rolling(18).sum() / v.rolling(18).sum().replace(0, np.nan)
    f["taker_buy_share"] = px.taker_base / v.replace(0, np.nan)
    f["demand_18"] = (2 * px.taker_base - v).rolling(18).sum() / v.rolling(18).sum().replace(0, np.nan)
    f["drawdown_360"] = c / c.rolling(360).max() - 1
    f["bounce_360"] = c / c.rolling(360).min() - 1
    f["bb_z"] = (c - c.rolling(42).mean()) / c.rolling(42).std().replace(0, np.nan)
    f["range_break_120"] = c / h.shift(1).rolling(120).max() - 1
    f["range_exit_60"] = c / l.shift(1).rolling(60).min() - 1
    # Price/volume proxies for crowding and panic, not measurements of mental states.
    f["down_share_18"] = (r < 0).rolling(18).mean()
    f["hour_sin"] = np.sin(2 * np.pi * px.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * px.index.hour / 24)
    f["weekend"] = (px.index.dayofweek >= 5).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)


def labels(px: pd.DataFrame, horizon: int) -> pd.Series:
    """Decision at close t; enter open t+1, exit open t+1+horizon."""
    return px.open.shift(-(horizon + 1)) / px.open.shift(-1) - 1


def volatility_scale(f: pd.DataFrame, cfg: dict) -> pd.Series:
    # Quantize to avoid tiny, costly resizing on every candle.
    scale = (cfg["target_annual_vol"] / f.vol_120.clip(lower=0.05)).clip(upper=cfg["max_sleeve_exposure"])
    return (np.floor(scale * 10) / 10).fillna(0).clip(lower=0, upper=cfg["max_sleeve_exposure"])


def stateful(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    out, active = [], False
    for e, x in zip(enter.fillna(False), exit_.fillna(False)):
        if active and x:
            active = False
        elif not active and e:
            active = True
        out.append(float(active))
    return pd.Series(out, index=enter.index)


def rules(px: pd.DataFrame, f: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = px.close
    p = pd.DataFrame(index=px.index)
    p["cash"] = 0.0
    for fast, slow in ((42, 180), (120, 360), (360, 1200)):
        p[f"trend_{fast}_{slow}"] = (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)
    p["momentum_120"] = (f.ret_120 > 0).astype(float)
    p["momentum_360"] = (f.ret_360 > 0).astype(float)
    p["breakout_120_60"] = stateful(f.range_break_120 > 0, f.range_exit_60 < 0)
    p["pullback_rsi2"] = stateful((f.dist_ma_360 > 0) & (f.rsi_2 < 0.1), f.dist_ma_18 > 0)
    p["bollinger_revert"] = stateful((f.dist_ma_360 > 0) & (f.bb_z < -2), f.bb_z > 0)
    p["demand_momentum"] = ((f.demand_18 > 0) & (f.ret_120 > 0) & (f.dist_ma_360 > 0)).astype(float)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    p["macd_trend"] = ((macd > macd.ewm(span=9, adjust=False).mean()) & (f.dist_ma_360 > 0)).astype(float)
    p = p.mul(volatility_scale(f, cfg), axis=0)
    p.iloc[:cfg["warmup_bars"]] = 0.0
    return p
