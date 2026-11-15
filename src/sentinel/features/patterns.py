"""Chart and candlestick patterns detected mathematically (no images, no lookahead).

Pattern flags use only the current and previous bars. Double bottoms/tops are found with a
causal swing-point scan: a swing low is confirmed only `k` bars after it happens."""
from __future__ import annotations

import numpy as np
import pandas as pd


def candlestick_patterns(df: pd.DataFrame, atr: pd.Series, lookback: int = 20) -> pd.DataFrame:
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - lo).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - lo
    po, pc = o.shift(1), c.shift(1)

    out = pd.DataFrame(index=df.index)
    out["bull_engulf"] = ((pc < po) & (c > o) & (o <= pc) & (c >= po)).astype(float)
    out["bear_engulf"] = ((pc > po) & (c < o) & (o >= pc) & (c <= po)).astype(float)
    out["hammer"] = ((lower >= 2 * body) & (upper <= body) & (body / rng > 0.1)).astype(float)
    out["shooting_star"] = ((upper >= 2 * body) & (lower <= body) & (body / rng > 0.1)).astype(float)
    out["doji"] = (body / rng < 0.1).fillna(False).astype(float)

    prev_hi = h.shift(1).rolling(lookback).max()
    prev_lo = lo.shift(1).rolling(lookback).min()
    out["breakout_20"] = (c > prev_hi).astype(float)
    out["breakdown_20"] = (c < prev_lo).astype(float)
    out["dist_high_20"] = (c / h.rolling(lookback).max() - 1).fillna(0.0)
    out["dist_low_20"] = (c / lo.rolling(lookback).min() - 1).fillna(0.0)

    db, dt = double_patterns(lo.to_numpy(), h.to_numpy(), c.to_numpy(), atr.to_numpy())
    out["double_bottom"] = db
    out["double_top"] = dt
    return out


def double_patterns(low: np.ndarray, high: np.ndarray, close: np.ndarray, atr: np.ndarray,
                    k: int = 3, window: int = 40, tol_atr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Flag bars where a double bottom (top) completes: two confirmed swing lows (highs) within
    `tol_atr` ATRs of each other inside `window` bars, and price closes back above (below) the
    intervening peak (trough) — the classic neckline break."""
    n = len(close)
    db, dt = np.zeros(n), np.zeros(n)
    swing_lows: list[int] = []
    swing_highs: list[int] = []
    for t in range(2 * k, n):
        p = t - k  # candidate pivot, confirmed now that k bars have passed
        lw = low[p - k:p + k + 1]
        hw = high[p - k:p + k + 1]
        if low[p] == lw.min():
            swing_lows.append(p)
        if high[p] == hw.max():
            swing_highs.append(p)
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue
        lows = [i for i in swing_lows[-4:] if t - i <= window]
        if len(lows) >= 2:
            i1, i2 = lows[-2], lows[-1]
            if abs(low[i1] - low[i2]) <= tol_atr * a and i2 - i1 >= k:
                neckline = high[i1:i2 + 1].max()
                if close[t] > neckline and close[t - 1] <= neckline:
                    db[t] = 1.0
        highs = [i for i in swing_highs[-4:] if t - i <= window]
        if len(highs) >= 2:
            i1, i2 = highs[-2], highs[-1]
            if abs(high[i1] - high[i2]) <= tol_atr * a and i2 - i1 >= k:
                neckline = low[i1:i2 + 1].min()
                if close[t] < neckline and close[t - 1] >= neckline:
                    dt[t] = 1.0
    return db, dt
