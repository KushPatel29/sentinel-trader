"""Historical bars and news from Alpaca's market-data REST APIs."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from sentinel.events import NewsItem
from sentinel.market_time import filter_regular_session

log = logging.getLogger(__name__)


def _feed(name: str):
    from alpaca.data.enums import DataFeed
    return DataFeed.SIP if name == "sip" else DataFeed.IEX


class AlpacaData:
    def __init__(self, api_key: str, secret_key: str, feed: str = "iex"):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.historical.news import NewsClient

        self.bars_client = StockHistoricalDataClient(api_key, secret_key)
        self.news_client = NewsClient(api_key, secret_key)
        self.feed = feed

    def fetch_bars(self, symbols: list[str], start: datetime, end: datetime, minutes: int) -> pd.DataFrame:
        """N-minute bars, split/dividend adjusted, regular session only. Chunked by 60 days."""
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        frames = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=60), end)
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
                start=cursor, end=chunk_end, adjustment=Adjustment.ALL, feed=_feed(self.feed),
            )
            df = self.bars_client.get_stock_bars(req).df
            if not df.empty:
                frames.append(df.reset_index())
            log.info("bars %s -> %s: %d rows", cursor.date(), chunk_end.date(), 0 if df.empty else len(df))
            cursor = chunk_end
        if not frames:
            return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        out = pd.concat(frames, ignore_index=True).rename(columns={"timestamp": "ts"})
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        out = out[["symbol", "ts", "open", "high", "low", "close", "volume"]]
        out = filter_regular_session(out, minutes)
        return out.drop_duplicates(["symbol", "ts"]).sort_values(["ts", "symbol"]).reset_index(drop=True)

    def recent_bars(self, symbols: list[str], minutes: int, n_bars: int) -> pd.DataFrame:
        """Enough recent history to warm up rolling features at startup."""
        trading_days = max(3, int(n_bars * minutes / 390) + 3)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(trading_days * 1.6) + 4)
        df = self.fetch_bars(symbols, start, end, minutes)
        return df.groupby("symbol", group_keys=False).tail(n_bars).reset_index(drop=True)

    def fetch_news(self, symbols: list[str], start: datetime, end: datetime) -> list[NewsItem]:
        from alpaca.data.requests import NewsRequest

        items: list[NewsItem] = []
        token = None
        while True:
            req = NewsRequest(symbols=",".join(symbols), start=start, end=end, limit=50,
                              include_content=False, page_token=token)
            res = self.news_client.get_news(req)
            batch = res.data.get("news", []) if hasattr(res, "data") else []
            for n in batch:
                items.append(news_from_alpaca(n))
            token = getattr(res, "next_page_token", None)
            if not token or not batch:
                break
        log.info("news %s -> %s: %d items", start.date(), end.date(), len(items))
        return items


def news_from_alpaca(n) -> NewsItem:
    """Works for both REST `News` models and websocket news objects."""
    ts = getattr(n, "created_at", None) or getattr(n, "updated_at", None)
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return NewsItem(
        id=str(n.id), ts=ts.astimezone(timezone.utc), headline=n.headline or "",
        summary=getattr(n, "summary", "") or "", symbols=tuple(getattr(n, "symbols", []) or ()),
        source=getattr(n, "source", "") or "",
    )
