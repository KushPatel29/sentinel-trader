"""Shared steps used by the CLI: load history, score news (cached), build the modelling table."""
from __future__ import annotations

import logging

import pandas as pd

from sentinel.config import Config
from sentinel.data.store import Store
from sentinel.events import NewsItem
from sentinel.models.dataset import build_dataset
from sentinel.models.sentiment import SentimentAnalyzer

log = logging.getLogger(__name__)


def load_history(cfg: Config, store: Store) -> tuple[pd.DataFrame, list[NewsItem]]:
    bars = store.load_bars(cfg.symbols, cfg.timeframe_minutes)
    if bars.empty:
        raise SystemExit("No bars in the database. Run `sentinel fetch` (or `sentinel fetch --synthetic`).")
    news = store.load_news(start=bars["ts"].min().to_pydatetime())
    missing = sorted(set(cfg.symbols) - set(bars["symbol"].unique()))
    if missing:
        log.warning("no bars stored for %s", missing)
    return bars, news


def score_news(store: Store, news: list[NewsItem], analyzer: SentimentAnalyzer) -> dict[str, float]:
    cached = store.load_sentiment(analyzer.engine)
    todo = [n for n in news if n.id not in cached]
    if todo:
        log.info("scoring %d headlines with %s", len(todo), analyzer.engine)
        for s in analyzer.score_many(todo):
            store.upsert_sentiment(s.news_id, s.engine, s.score)
            cached[s.news_id] = s.score
    return {n.id: cached[n.id] for n in news if n.id in cached}


def make_dataset(cfg: Config, store: Store, analyzer: SentimentAnalyzer):
    bars, news = load_history(cfg, store)
    scores = score_news(store, news, analyzer)
    df = build_dataset(bars, news, scores, cfg)
    return df, bars, news, scores
