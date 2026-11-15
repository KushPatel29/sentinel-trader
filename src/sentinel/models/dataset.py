"""Turns bars + news into a leak-free modelling table."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from sentinel.config import Config
from sentinel.events import NewsItem
from sentinel.features.labels import triple_barrier
from sentinel.features.technical import TECH_FEATURES, WARMUP_BARS, compute_features
from sentinel.market_time import NY
from sentinel.models.sentiment import SentimentState

SENT_FEATURES = ["sent_score", "sent_intensity"]
MODEL_FEATURES = TECH_FEATURES + SENT_FEATURES + ["regime"]


def sentiment_asof(rows: pd.DataFrame, news: list[NewsItem], scores: dict[str, float], decay_minutes: float,
                   minutes: int) -> pd.DataFrame:
    """Decayed sentiment for each (symbol, bar end) using only headlines published by the bar's end."""
    state = SentimentState(decay_minutes)
    tmp = pd.DataFrame({"end": (rows["ts"] + timedelta(minutes=minutes)).to_numpy(),
                        "symbol": rows["symbol"].to_numpy(), "pos": np.arange(len(rows))}
                       ).sort_values("end", kind="stable")
    tmp["end"] = pd.to_datetime(tmp["end"], utc=True)
    news_sorted = sorted((n for n in news if n.id in scores), key=lambda n: n.ts)
    s_out = np.zeros(len(rows))
    i_out = np.zeros(len(rows))
    j = 0
    for idx, end, sym in zip(tmp["pos"], tmp["end"], tmp["symbol"]):
        t = end.to_pydatetime()
        while j < len(news_sorted) and news_sorted[j].ts <= t:
            n = news_sorted[j]
            state.add(n.symbols, n.ts, scores[n.id])
            j += 1
        s_out[idx] = state.score(sym, t)
        i_out[idx] = state.intensity(sym, t)
    return pd.DataFrame({"sent_score": s_out, "sent_intensity": i_out}, index=rows.index)


def entry_window_mask(ts: pd.Series, cfg: Config) -> pd.Series:
    ny = ts.dt.tz_convert(NY)
    start_min = ny.dt.hour * 60 + ny.dt.minute - (9 * 60 + 30)
    end_min = start_min + cfg.timeframe_minutes            # decision is made at the bar's end
    return (end_min >= cfg.risk.no_entry_minutes_after_open) & (390 - end_min >= cfg.risk.no_entry_minutes_before_close)


def build_dataset(bars: pd.DataFrame, news: list[NewsItem], scores: dict[str, float], cfg: Config) -> pd.DataFrame:
    parts = []
    for _, g in bars.groupby("symbol", sort=True):
        g = g.sort_values("ts").reset_index(drop=True)
        f = compute_features(g)
        f["label"] = triple_barrier(g, f["atr"], cfg.labels.horizon_bars, cfg.labels.tp_atr, cfg.labels.sl_atr)
        f["high"], f["low"], f["open"] = g["high"], g["low"], g["open"]
        parts.append(f.iloc[WARMUP_BARS:])
    df = pd.concat(parts, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)
    df = pd.concat([df, sentiment_asof(df, news, scores, cfg.sentiment.decay_minutes, cfg.timeframe_minutes)], axis=1)
    df["entry_window"] = entry_window_mask(df["ts"], cfg)
    return df
