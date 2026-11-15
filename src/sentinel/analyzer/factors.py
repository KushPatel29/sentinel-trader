"""Daily factor library, grouped by what moves prices.

Every factor at date t uses data up to and including t's close (no lookahead). Context series
(market, VIX, rates, dollar) are forward-filled onto the asset's calendar, so crypto's weekends
carry the previous Friday's values."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS: dict[str, list[str]] = {
    "Trend": ["dist_sma20", "dist_sma50", "dist_sma200", "sma50_vs_200", "sma50_slope", "adx14", "macd_hist"],
    "Momentum": ["ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1"],
    "Mean reversion": ["rsi14", "rsi2", "bb_pos", "zscore20", "ret_1"],
    "Volatility": ["vol21", "vol63", "vol_ratio", "atr_pct", "vol_pctile"],
    "Demand & volume": ["volume_z", "obv_slope", "ad_slope", "mfi14", "updown_vol"],
    "Psychology & sentiment": ["pos_52w", "drawdown", "streak", "gap", "vix", "vix_pctile", "vix_chg5",
                               "greed_index"],
    "Macro & cross-asset": ["mkt_ret_21", "mkt_trend", "rel_strength_63", "beta63", "corr63", "rates_chg21",
                            "dollar_ret21", "crypto_mkt_ret21"],
    "Seasonality": ["dow_sin", "dow_cos", "month_sin", "month_cos", "turn_of_month"],
}
ALL_FACTORS = [f for g in GROUPS.values() for f in g]
WARMUP = 252


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _slope(s: pd.Series, n: int) -> pd.Series:
    """Rolling least-squares slope per bar."""
    x = np.arange(n) - (n - 1) / 2
    return s.rolling(n).apply(lambda y: float(np.dot(x, y) / np.dot(x, x)), raw=True)


def _pctile(s: pd.Series, n: int = 252) -> pd.Series:
    return s.rolling(n, min_periods=60).rank(pct=True)


def compute_factors(px: pd.DataFrame, ctx: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    ctx = ctx or {}
    o, h, lo, c, v = px["open"], px["high"], px["low"], px["close"], px["volume"]
    r1 = np.log(c).diff()
    f = pd.DataFrame(index=px.index)

    # ---- trend
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    f["dist_sma20"], f["dist_sma50"], f["dist_sma200"] = c / sma20 - 1, c / sma50 - 1, c / sma200 - 1
    f["sma50_vs_200"] = sma50 / sma200 - 1
    f["sma50_slope"] = sma50.pct_change(20)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    up_move, dn_move = h.diff(), -lo.diff()
    pdm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=px.index)
    ndm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=px.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    f["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean() * np.sign(pdi - ndi)   # signed trend strength
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    # ---- momentum
    for k in (5, 21, 63, 126, 252):
        f[f"ret_{k}"] = c.pct_change(k)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1

    # ---- mean reversion
    f["rsi14"], f["rsi2"] = _rsi(c, 14), _rsi(c, 2)
    sd20 = c.rolling(20).std()
    f["bb_pos"] = (c - sma20) / (2 * sd20)
    f["zscore20"] = (c - sma20) / sd20
    f["ret_1"] = c.pct_change()

    # ---- volatility
    f["vol21"] = r1.rolling(21).std() * np.sqrt(252)
    f["vol63"] = r1.rolling(63).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol21"] / f["vol63"]
    f["atr_pct"] = atr / c
    f["vol_pctile"] = _pctile(f["vol21"])

    # ---- demand & volume (NaN when the instrument has no volume, e.g. indices/FX)
    has_vol = v.rolling(20).sum() > 0
    lv = np.log1p(v)
    f["volume_z"] = ((lv - lv.rolling(20).mean()) / lv.rolling(20).std()).where(has_vol)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_slope"] = (_slope(obv, 20) / v.rolling(20).mean()).where(has_vol)          # buying vs selling pressure
    clv = ((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)
    ad = (clv.fillna(0) * v).cumsum()
    f["ad_slope"] = (_slope(ad, 20) / v.rolling(20).mean()).where(has_vol)            # accumulation/distribution
    tp = (h + lo + c) / 3
    mf = tp * v
    pos_mf = mf.where(tp > tp.shift(), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(), 0.0).rolling(14).sum()
    f["mfi14"] = (100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))).where(has_vol)
    upv = v.where(c > c.shift(), 0.0).rolling(20).sum()
    dnv = v.where(c < c.shift(), 0.0).rolling(20).sum()
    f["updown_vol"] = (np.log((upv + 1) / (dnv + 1))).where(has_vol)

    # ---- psychology: anchoring, fear/greed, herding
    hi52, lo52 = h.rolling(252, min_periods=60).max(), lo.rolling(252, min_periods=60).min()
    f["pos_52w"] = (c - lo52) / (hi52 - lo52)
    f["drawdown"] = c / c.cummax() - 1
    sign = np.sign(c.diff()).fillna(0)
    grp = (sign != sign.shift()).cumsum()
    f["streak"] = sign.groupby(grp).cumcount().add(1) * sign
    f["gap"] = o / c.shift() - 1

    def ctx_close(key: str) -> pd.Series | None:
        d = ctx.get(key)
        if d is None or d.empty:
            return None
        return d["close"].reindex(px.index.union(d.index)).ffill().reindex(px.index)

    vix = ctx_close("vix")
    if vix is not None:
        f["vix"] = vix
        f["vix_pctile"] = _pctile(vix)
        f["vix_chg5"] = vix.pct_change(5)
    else:
        f["vix"] = f["vix_pctile"] = f["vix_chg5"] = np.nan

    # ---- macro & cross-asset
    mkt = ctx_close("market")
    if mkt is not None:
        mr = mkt.pct_change()
        f["mkt_ret_21"] = mkt.pct_change(21)
        f["mkt_trend"] = mkt / mkt.rolling(200).mean() - 1
        f["rel_strength_63"] = f["ret_63"] - mkt.pct_change(63)
        ar = c.pct_change()
        f["beta63"] = ar.rolling(63).cov(mr) / mr.rolling(63).var()
        f["corr63"] = ar.rolling(63).corr(mr)
    else:
        for k in ("mkt_ret_21", "mkt_trend", "rel_strength_63", "beta63", "corr63"):
            f[k] = np.nan
    rates, dollar, cm = ctx_close("rates"), ctx_close("dollar"), ctx_close("crypto_market")
    f["rates_chg21"] = rates.diff(21) if rates is not None else np.nan
    f["dollar_ret21"] = dollar.pct_change(21) if dollar is not None else np.nan
    f["crypto_mkt_ret21"] = cm.pct_change(21) if cm is not None else np.nan

    # Greed index (0 = extreme fear, 100 = extreme greed), CNN-style composite of what is available:
    # momentum vs 125-day average, 52-week position, inverse volatility percentile, VIX percentile.
    comps = [
        (c / c.rolling(125).mean() - 1).rolling(252, min_periods=60).rank(pct=True),
        f["pos_52w"],
        1 - f["vol_pctile"],
    ]
    if vix is not None:
        comps.append(1 - f["vix_pctile"])
    f["greed_index"] = pd.concat(comps, axis=1).mean(axis=1) * 100

    # ---- seasonality
    dow, mon = px.index.dayofweek, px.index.month
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f["month_sin"], f["month_cos"] = np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12)
    dom = px.index.day
    f["turn_of_month"] = ((dom <= 3) | (dom >= 27)).astype(float)

    f["close"] = c
    return f.replace([np.inf, -np.inf], np.nan)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1
