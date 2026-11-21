"""Pattern recognition for gold: classic chart patterns, candlestick patterns and analog (look-alike) matching.

Every pattern is dated on the bar where it becomes KNOWN (confirmation), never earlier, so tests have no
lookahead. Each pattern implies a direction: +1 (buy) or -1 (sell).

Chart patterns use swing points from a causal zig-zag: a swing high is confirmed only once price has fallen
`k` ATRs below it (and vice versa), which is exactly when a chartist could have drawn it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PATTERN_INFO = {
    "double_bottom": ("Double bottom", +1, "Two similar lows, then a close above the peak between them"),
    "double_top": ("Double top", -1, "Two similar highs, then a close below the trough between them"),
    "inv_head_shoulders": ("Inverse head & shoulders", +1, "Low, lower low, similar low, then a neckline break up"),
    "head_shoulders": ("Head & shoulders", -1, "High, higher high, similar high, then a neckline break down"),
    "triangle_up": ("Triangle breakout up", +1, "Converging swing highs and lows, then a close above the upper line"),
    "triangle_down": ("Triangle breakout down", -1, "Converging swing highs and lows, then a close below the lower line"),
    "range_break_up": ("Range breakout up", +1, "Unusually tight 20-bar range, then a close above it"),
    "range_break_down": ("Range breakout down", -1, "Unusually tight 20-bar range, then a close below it"),
    "bull_flag": ("Bull flag", +1, "Sharp rise, tight pause, then a close above the pause"),
    "bear_flag": ("Bear flag", -1, "Sharp fall, tight pause, then a close below the pause"),
    "bull_engulfing": ("Bullish engulfing", +1, "After a decline, an up candle engulfs the prior down candle"),
    "bear_engulfing": ("Bearish engulfing", -1, "After a rise, a down candle engulfs the prior up candle"),
    "hammer": ("Hammer", +1, "After a decline, long lower wick and small body near the high"),
    "shooting_star": ("Shooting star", -1, "After a rise, long upper wick and small body near the low"),
    "morning_star": ("Morning star", +1, "Down candle, small candle, strong up candle closing into the first"),
    "evening_star": ("Evening star", -1, "Up candle, small candle, strong down candle closing into the first"),
    "three_soldiers": ("Three white soldiers", +1, "Three strong up candles in a row"),
    "three_crows": ("Three black crows", -1, "Three strong down candles in a row"),
    "inside_break_up": ("Inside-bar breakout up", +1, "A bar inside the previous one, then a close above the mother bar"),
    "inside_break_down": ("Inside-bar breakout down", -1, "A bar inside the previous one, then a close below the mother bar"),
}


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


@dataclass
class Pivot:
    i: int           # bar of the extreme
    price: float
    kind: str        # "H" or "L"
    confirm: int     # bar on which it became known


def zigzag(df: pd.DataFrame, k: float = 2.0) -> list[Pivot]:
    """Causal zig-zag. A swing high is confirmed on the first bar whose low is k ATRs below the running high
    since the last swing low (and vice versa). `confirm` is that bar: the earliest moment it is knowable."""
    h, lo, a = df["high"].to_numpy(), df["low"].to_numpy(), atr(df).to_numpy()
    piv: list[Pivot] = []
    direction = 0                                 # 0 = undecided, +1 = tracking a high, -1 = tracking a low
    hi_i = lo_i = 0
    for i in range(1, len(h)):
        if not np.isfinite(a[i]):
            hi_i = i if h[i] >= h[hi_i] else hi_i
            lo_i = i if lo[i] <= lo[lo_i] else lo_i
            continue
        if direction == 0:
            if h[i] >= h[hi_i]:
                hi_i = i
            if lo[i] <= lo[lo_i]:
                lo_i = i
            if h[i] - lo[lo_i] >= k * a[i] and lo_i < i:
                piv.append(Pivot(lo_i, lo[lo_i], "L", i))
                direction, hi_i = 1, i
            elif h[hi_i] - lo[i] >= k * a[i] and hi_i < i:
                piv.append(Pivot(hi_i, h[hi_i], "H", i))
                direction, lo_i = -1, i
        elif direction == 1:
            if h[i] >= h[hi_i]:
                hi_i = i
            elif h[hi_i] - lo[i] >= k * a[i]:
                piv.append(Pivot(hi_i, h[hi_i], "H", i))
                direction, lo_i = -1, i
        else:
            if lo[i] <= lo[lo_i]:
                lo_i = i
            elif h[i] - lo[lo_i] >= k * a[i]:
                piv.append(Pivot(lo_i, lo[lo_i], "L", i))
                direction, hi_i = 1, i
    return piv


def chart_patterns(df: pd.DataFrame, k: float = 2.0, window: int = 20) -> pd.DataFrame:
    """Signals (index = detection bar) for pivot-based patterns and range/flag breakouts."""
    c = df["close"].to_numpy()
    a = atr(df).to_numpy()
    n = len(c)
    sig = {key: np.zeros(n) for key in ("double_bottom", "double_top", "inv_head_shoulders", "head_shoulders",
                                         "triangle_up", "triangle_down", "range_break_up", "range_break_down",
                                         "bull_flag", "bear_flag")}
    piv = zigzag(df, k)
    known: list[Pivot] = []
    pi = 0
    armed: dict[str, tuple[float, int]] = {}     # pattern -> (trigger level, expiry bar)
    for t in range(n):
        while pi < len(piv) and piv[pi].confirm <= t:
            known.append(piv[pi])
            pi += 1
            p = known[-4:]
            at = a[t] if np.isfinite(a[t]) else np.nan
            if len(p) >= 3 and np.isfinite(at):
                x1, x2, x3 = p[-3], p[-2], p[-1]
                if x1.kind == "L" and x2.kind == "H" and x3.kind == "L" and abs(x1.price - x3.price) <= 0.75 * at:
                    armed["double_bottom"] = (x2.price, t + window)
                if x1.kind == "H" and x2.kind == "L" and x3.kind == "H" and abs(x1.price - x3.price) <= 0.75 * at:
                    armed["double_top"] = (x2.price, t + window)
            if len(known) >= 5 and np.isfinite(at):
                q = known[-5:]
                if [z.kind for z in q] == ["L", "H", "L", "H", "L"]:
                    ls, hd, rs = q[0].price, q[2].price, q[4].price
                    if hd < ls - 0.5 * at and hd < rs - 0.5 * at and abs(ls - rs) <= 1.0 * at:
                        armed["inv_head_shoulders"] = (max(q[1].price, q[3].price), t + window)
                if [z.kind for z in q] == ["H", "L", "H", "L", "H"]:
                    ls, hd, rs = q[0].price, q[2].price, q[4].price
                    if hd > ls + 0.5 * at and hd > rs + 0.5 * at and abs(ls - rs) <= 1.0 * at:
                        armed["head_shoulders"] = (min(q[1].price, q[3].price), t + window)
            if len(known) >= 4 and np.isfinite(at):
                hs = [z for z in known[-4:] if z.kind == "H"]
                ls_ = [z for z in known[-4:] if z.kind == "L"]
                if len(hs) == 2 and len(ls_) == 2:
                    falling_highs = hs[1].price <= hs[0].price + 0.25 * at
                    rising_lows = ls_[1].price >= ls_[0].price - 0.25 * at
                    squeeze = (hs[0].price - hs[1].price) + (ls_[1].price - ls_[0].price)
                    if falling_highs and rising_lows and squeeze > 0.5 * at:
                        def line(p0, p1):
                            slope = (p1.price - p0.price) / max(1, p1.i - p0.i)
                            return p1.price, slope, p1.i
                        armed["triangle_up"] = (line(hs[0], hs[1]), t + window)
                        armed["triangle_down"] = (line(ls_[0], ls_[1]), t + window)
        for key, (lvl, exp) in list(armed.items()):
            if t > exp:
                armed.pop(key)
                continue
            if key.startswith("triangle"):
                p1, slope, i1 = lvl
                level = p1 + slope * (t - i1)
            else:
                level = lvl
            prev = c[t - 1] if t else c[t]
            if key in ("double_bottom", "inv_head_shoulders", "triangle_up") and c[t] > level >= prev:
                sig[key][t] = 1
                armed.pop(key)
                if key == "triangle_up":
                    armed.pop("triangle_down", None)
            elif key in ("double_top", "head_shoulders", "triangle_down") and c[t] < level <= prev:
                sig[key][t] = 1
                armed.pop(key)
                if key == "triangle_down":
                    armed.pop("triangle_up", None)

    s = pd.DataFrame(sig, index=df.index)
    # range breakout: tight 20-bar range relative to its own history, then a close beyond it
    hi20 = df["high"].shift(1).rolling(20).max()
    lo20 = df["low"].shift(1).rolling(20).min()
    width = (hi20 - lo20) / df["close"]
    tight = width < width.rolling(250, min_periods=100).quantile(0.25)
    s["range_break_up"] = (tight & (df["close"] > hi20)).astype(float)
    s["range_break_down"] = (tight & (df["close"] < lo20)).astype(float)
    # flags: a strong 10-bar move ("pole"), a 5-bar pause with a small range, then a close beyond the pause
    sig_d = np.log(df["close"]).diff().rolling(60).std()
    pole = np.log(df["close"].shift(5) / df["close"].shift(15))
    p_hi, p_lo = df["high"].shift(1).rolling(5).max(), df["low"].shift(1).rolling(5).min()
    pause_small = (p_hi - p_lo) / df["close"] < 0.6 * pole.abs()
    strong = pole.abs() > 2.0 * sig_d * np.sqrt(10)
    s["bull_flag"] = (strong & (pole > 0) & pause_small & (df["close"] > p_hi)).astype(float)
    s["bear_flag"] = (strong & (pole < 0) & pause_small & (df["close"] < p_lo)).astype(float)
    return s


def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - lo).replace(0, np.nan)
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - lo
    a = atr(df)
    down = c.shift(1) < c.shift(1).rolling(10).mean()            # prior decline
    up = c.shift(1) > c.shift(1).rolling(10).mean()              # prior rise
    po, pc = o.shift(1), c.shift(1)
    s = pd.DataFrame(index=df.index)
    s["bull_engulfing"] = (down & (pc < po) & (c > o) & (o <= pc) & (c >= po)).astype(float)
    s["bear_engulfing"] = (up & (pc > po) & (c < o) & (o >= pc) & (c <= po)).astype(float)
    s["hammer"] = (down & (lower >= 2 * body) & (upper <= 0.3 * body + 0.1 * rng)).astype(float)
    s["shooting_star"] = (up & (upper >= 2 * body) & (lower <= 0.3 * body + 0.1 * rng)).astype(float)
    big = body > 0.6 * a
    small1 = body.shift(1) < 0.3 * a
    s["morning_star"] = (down.shift(1).fillna(False) & big.shift(2).fillna(False) & (c.shift(2) < o.shift(2)) & small1
                         & (c > o) & big & (c > (o.shift(2) + c.shift(2)) / 2)).astype(float)
    s["evening_star"] = (up.shift(1).fillna(False) & big.shift(2).fillna(False) & (c.shift(2) > o.shift(2)) & small1
                         & (c < o) & big & (c < (o.shift(2) + c.shift(2)) / 2)).astype(float)
    ups = (c > o) & (body > 0.5 * a) & (c > c.shift(1))
    dns = (c < o) & (body > 0.5 * a) & (c < c.shift(1))
    s["three_soldiers"] = (ups & ups.shift(1).fillna(False) & ups.shift(2).fillna(False)).astype(float)
    s["three_crows"] = (dns & dns.shift(1).fillna(False) & dns.shift(2).fillna(False)).astype(float)
    inside = (h.shift(1) <= h.shift(2)) & (lo.shift(1) >= lo.shift(2))
    s["inside_break_up"] = (inside & (c > h.shift(2))).astype(float)
    s["inside_break_down"] = (inside & (c < lo.shift(2))).astype(float)
    return s


def all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([chart_patterns(df), candle_patterns(df)], axis=1).fillna(0.0)


# ---------------------------------------------------------------------------------------------- analogs
def analog_forecast(close: pd.Series, horizon: int = 5, window: int = 20, k: int = 50, min_history: int = 1000,
                    start: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """For each day, find the k past windows whose (volatility-normalised) price path looked most like the last
    `window` bars, and average what happened over the next `horizon` bars. Only windows whose outcome was already
    known are used."""
    r = np.log(close).diff()
    vol = r.rolling(60).std()
    z = (r / vol).to_numpy()
    fwd = np.log(close.shift(-horizon) / close).to_numpy()
    n = len(close)
    X = np.full((n, window), np.nan)
    for j in range(window):
        X[:, window - 1 - j] = np.roll(z, j)
    X[:window + 60] = np.nan
    valid = np.isfinite(X).all(axis=1)
    out_p, out_m, idx = [], [], []
    first = max(min_history, window + 60 + horizon)
    if start is not None:
        first = max(first, int(np.searchsorted(close.index.values, np.datetime64(pd.Timestamp(start)))))
    for t in range(first, n):
        if not valid[t]:
            continue
        hist = np.arange(window + 60, t - horizon)            # outcomes fully known by t
        hist = hist[valid[hist] & np.isfinite(fwd[hist])]
        if len(hist) < k:
            continue
        d = np.sqrt(((X[hist] - X[t]) ** 2).sum(axis=1))
        nn = hist[np.argpartition(d, k)[:k]]
        out_p.append(float((fwd[nn] > 0).mean()))
        out_m.append(float(fwd[nn].mean()))
        idx.append(close.index[t])
    return pd.DataFrame({"analog_up_share": out_p, "analog_mean_return": out_m}, index=idx)
