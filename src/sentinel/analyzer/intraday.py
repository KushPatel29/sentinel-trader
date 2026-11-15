"""Intraday lab: the Sentinel bot's 15-minute strategy on the last ~60 days of real bars.

1. Walk-forward accuracy + event-driven backtest on out-of-sample predictions.
2. Forward simulation: a model trained only on data before the last N sessions trades those
   sessions bar by bar through the live code path (rolling features, per-bar inference, risk officer,
   simulated bracket orders) - what the paper bot would have done."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

from sentinel.backtest.engine import BacktestResult, run_simulation
from sentinel.config import Config
from sentinel.features.technical import compute_all
from sentinel.models.dataset import build_dataset
from sentinel.models.gbm import train_final, walk_forward
from sentinel.models.sentiment import SentimentState
from sentinel.strategy.signals import WINDOW, LiveSignals, PrecomputedSignals


@dataclass
class IntradayLab:
    wf_metrics: pd.DataFrame
    auc: float
    base_rate: float
    backtest: BacktestResult
    forward: BacktestResult
    forward_start: pd.Timestamp
    n_sessions: int


def lab_config(symbols: list[str], n_timestamps: int) -> Config:
    cfg = Config(symbols=symbols, timeframe_minutes=15)
    cfg.model.min_train_bars = max(300, int(n_timestamps * 0.45))
    cfg.model.test_bars = max(100, int(n_timestamps * 0.12))
    cfg.model.lgbm = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    return cfg


def run_lab(bars: pd.DataFrame, forward_sessions: int = 5) -> IntradayLab:
    logging.getLogger("sentinel.engine").setLevel(logging.WARNING)
    logging.getLogger("sentinel.risk.officer").setLevel(logging.WARNING)
    symbols = sorted(bars["symbol"].unique())
    cfg = lab_config(symbols, bars["ts"].nunique())

    df = build_dataset(bars, [], {}, cfg)
    oos, wf = walk_forward(df, cfg)
    ev = df.merge(oos, on=["ts", "symbol"])
    ev = ev[ev["entry_window"] & ev["label"].notna()]
    auc = float(roc_auc_score(ev["label"], ev["prob"])) if ev["label"].nunique() == 2 else 0.5
    feats = compute_all(bars)
    sim = bars[(bars["ts"] >= oos["ts"].min()) & (bars["ts"] <= oos["ts"].max())]
    bt = asyncio.run(run_simulation(cfg, sim, [], PrecomputedSignals(oos, feats, cfg), scores={}))

    # forward simulation over the last N sessions with a model that never saw them
    days = bars["ts"].dt.tz_convert("America/New_York").dt.date
    last_days = sorted(days.unique())[-forward_sessions:]
    fstart = bars.loc[days == last_days[0], "ts"].min()
    bundle = train_final(df[df["ts"] < fstart - pd.Timedelta(hours=1)], cfg)
    state = SentimentState(cfg.sentiment.decay_minutes)
    live = LiveSignals(bundle, state, cfg)
    live.warm_up(bars[bars["ts"] < fstart].groupby("symbol").tail(WINDOW))
    fwd = asyncio.run(run_simulation(cfg, bars[bars["ts"] >= fstart], [], live, sentiment_state=state))
    return IntradayLab(wf, auc, float(ev["label"].mean()), bt, fwd, fstart, len(last_days))
