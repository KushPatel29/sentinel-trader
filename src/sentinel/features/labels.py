"""Triple-barrier labels that mirror the bot's real bracket orders.

For a signal at the close of bar t (entry ~ close_t):
  upper = close_t + tp_atr * ATR_t,  lower = close_t - sl_atr * ATR_t
label = 1 if the upper barrier is touched before the lower one within `horizon` bars, else 0.
If both are touched inside the same bar we assume the stop hit first (conservative).
The scan never crosses the session close, because the bot flattens before the close."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.market_time import session_id


def triple_barrier(df: pd.DataFrame, atr: pd.Series, horizon: int, tp_atr: float, sl_atr: float) -> pd.Series:
    """df: one symbol sorted by ts. Returns float labels (NaN where undefined: no ATR or no future bars)."""
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    a = atr.to_numpy()
    sess = session_id(df["ts"]).to_numpy()
    n = len(c)
    upper = c + tp_atr * a
    lower = c - sl_atr * a
    label = np.full(n, np.nan)
    resolved = np.zeros(n, dtype=bool)
    has_future = np.zeros(n, dtype=bool)
    for k in range(1, horizon + 1):
        idx = np.arange(n) + k
        valid = idx < n
        idx_c = np.where(valid, idx, n - 1)
        same_sess = valid & (sess[idx_c] == sess)
        has_future |= same_sess
        live = same_sess & ~resolved & np.isfinite(a)
        hit_sl = live & (lo[idx_c] <= lower)
        hit_tp = live & ~hit_sl & (h[idx_c] >= upper)
        label[hit_sl] = 0.0
        label[hit_tp] = 1.0
        resolved |= hit_sl | hit_tp
    timeout = has_future & ~resolved & np.isfinite(a)
    label[timeout] = 0.0
    return pd.Series(label, index=df.index)
