"""Causal technical features for one symbol's bars.

Every value at row t uses only bars <= t (the current bar is complete when the bot decides).
Features are scale-free (returns, ratios, z-scores) so one model can be pooled across symbols."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.features.patterns import candlestick_patterns
from sentinel.market_time import NY

WARMUP_BARS = 120  # rows needed before features are trustworthy


def wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
                   axis=1).max(axis=1)
    return wilder(tr, n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    down = wilder((-d).clip(lower=0), n)
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: one symbol, columns ts/open/high/low/close/volume, sorted by ts. Returns feature frame."""
    df = df.reset_index(drop=True)
    c, o, h, lo, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    logc = np.log(c)
    r1 = logc.diff()
    out = pd.DataFrame({"symbol": df["symbol"], "ts": df["ts"]})

    for k in (1, 3, 6, 12, 26):
        out[f"ret_{k}"] = logc.diff(k)
    out["vol_12"] = r1.rolling(12).std()
    out["vol_48"] = r1.rolling(48).std()
    out["vol_ratio"] = out["vol_12"] / out["vol_48"]

    a = atr(df, 14)
    out["atr"] = a                       # raw ATR (price units) for sizing/labels, not a model input
    out["atr_pct"] = a / c
    out["rsi_14"] = rsi(c, 14) / 100.0

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_pos"] = (c - ma20) / (2 * sd20)
    out["dist_ma20"] = c / ma20 - 1
    out["dist_ma50"] = c / c.rolling(50).mean() - 1

    lv = np.log1p(v)
    out["volume_z"] = (lv - lv.rolling(48).mean()) / lv.rolling(48).std()
    out["range_pct"] = (h - lo) / c
    out["close_loc"] = ((c - lo) / (h - lo)).fillna(0.5)

    ny = df["ts"].dt.tz_convert(NY)
    mins = (ny.dt.hour * 60 + ny.dt.minute) - (9 * 60 + 30)
    out["tod"] = mins / 390.0
    day = ny.dt.date
    day_open = o.groupby(day).transform("first")
    prev_close = c.groupby(day).transform("last").shift(1)
    # previous session's close, broadcast within each day
    prev_day_close = prev_close.where(day != day.shift(1)).ffill()
    out["gap"] = (day_open / prev_day_close - 1).fillna(0.0)
    out["day_ret"] = c / day_open - 1

    out = pd.concat([out, candlestick_patterns(df, a)], axis=1)
    out["close"] = c
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


TECH_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_26", "vol_12", "vol_48", "vol_ratio", "atr_pct", "rsi_14",
    "macd_hist", "bb_pos", "dist_ma20", "dist_ma50", "volume_z", "range_pct", "close_loc", "tod", "gap",
    "day_ret", "bull_engulf", "bear_engulf", "hammer", "shooting_star", "doji", "breakout_20", "breakdown_20",
    "double_bottom", "double_top", "dist_high_20", "dist_low_20",
]


def compute_all(bars: pd.DataFrame) -> pd.DataFrame:
    """Features for many symbols (pooled), sorted by ts then symbol."""
    parts = [compute_features(g.sort_values("ts")) for _, g in bars.groupby("symbol", sort=True)]
    return pd.concat(parts, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)
