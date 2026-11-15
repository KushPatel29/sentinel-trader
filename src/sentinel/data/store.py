"""SQLite persistence: market data, news, and a full audit trail of what the bot decided and did.

SQLite in WAL mode handles one writer (the bot) plus readers (the dashboard) comfortably at this
bar frequency. All writes go through one lock so the store is safe to share across threads."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sentinel.events import Bar, NewsItem

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL, ts TEXT NOT NULL, minutes INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, ts, minutes)
);
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, symbols TEXT, headline TEXT, summary TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS sentiment (
    news_id TEXT, engine TEXT, score REAL, rationale TEXT, PRIMARY KEY (news_id, engine)
);
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT, symbol TEXT, prob REAL, regime INTEGER, sentiment REAL, action TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    ts TEXT, client_order_id TEXT, symbol TEXT, side TEXT, qty REAL, status TEXT,
    filled_price REAL, take_profit REAL, stop_loss REAL, extra TEXT
);
CREATE TABLE IF NOT EXISTS fills (
    ts TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, reason TEXT, pnl REAL
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT, equity REAL, cash REAL, gross_exposure REAL, open_positions INTEGER
);
CREATE TABLE IF NOT EXISTS risk_events (
    ts TEXT, level TEXT, rule TEXT, message TEXT
);
CREATE INDEX IF NOT EXISTS ix_news_ts ON news(ts);
CREATE INDEX IF NOT EXISTS ix_signals_ts ON signals(ts);
"""


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        if str(path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, rows: list[tuple] | tuple = ()) -> None:
        with self._lock:
            if isinstance(rows, list):
                self._conn.executemany(sql, rows)
            else:
                self._conn.execute(sql, rows)
            self._conn.commit()

    # ---------- market data ----------
    def upsert_bars_df(self, df: pd.DataFrame, minutes: int) -> int:
        if df.empty:
            return 0
        rows = [
            (r.symbol, _iso(r.ts.to_pydatetime()), minutes, r.open, r.high, r.low, r.close, r.volume)
            for r in df.itertuples(index=False)
        ]
        self._exec("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def upsert_bar(self, bar: Bar) -> None:
        self._exec(
            "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)",
            (bar.symbol, _iso(bar.ts), bar.minutes, bar.open, bar.high, bar.low, bar.close, bar.volume),
        )

    def load_bars(self, symbols: list[str], minutes: int, start: datetime | None = None,
                  end: datetime | None = None) -> pd.DataFrame:
        q = "SELECT symbol, ts, open, high, low, close, volume FROM bars WHERE minutes=? AND symbol IN (%s)" % (
            ",".join("?" * len(symbols)))
        params: list = [minutes, *symbols]
        if start is not None:
            q += " AND ts >= ?"
            params.append(_iso(start))
        if end is not None:
            q += " AND ts < ?"
            params.append(_iso(end))
        with self._lock:
            df = pd.read_sql_query(q + " ORDER BY ts, symbol", self._conn, params=params)
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df

    # ---------- news ----------
    def upsert_news(self, items: list[NewsItem]) -> int:
        rows = [(n.id, _iso(n.ts), ",".join(n.symbols), n.headline, n.summary, n.source) for n in items]
        if rows:
            self._exec("INSERT OR REPLACE INTO news VALUES (?,?,?,?,?,?)", rows)
        return len(rows)

    def load_news(self, start: datetime | None = None, end: datetime | None = None) -> list[NewsItem]:
        q, params = "SELECT id, ts, symbols, headline, summary, source FROM news WHERE 1=1", []
        if start is not None:
            q += " AND ts >= ?"
            params.append(_iso(start))
        if end is not None:
            q += " AND ts < ?"
            params.append(_iso(end))
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY ts", params).fetchall()
        return [
            NewsItem(id=r[0], ts=datetime.fromisoformat(r[1]), symbols=tuple(s for s in (r[2] or "").split(",") if s),
                     headline=r[3] or "", summary=r[4] or "", source=r[5] or "")
            for r in rows
        ]

    def upsert_sentiment(self, news_id: str, engine: str, score: float, rationale: str = "") -> None:
        self._exec("INSERT OR REPLACE INTO sentiment VALUES (?,?,?,?)", (news_id, engine, score, rationale))

    def load_sentiment(self, engine: str) -> dict[str, float]:
        with self._lock:
            rows = self._conn.execute("SELECT news_id, score FROM sentiment WHERE engine=?", (engine,)).fetchall()
        return {r[0]: r[1] for r in rows}

    # ---------- audit trail ----------
    def log_signal(self, ts: datetime, symbol: str, prob: float, regime: int, sentiment: float,
                   action: str, reason: str) -> None:
        self._exec("INSERT INTO signals VALUES (?,?,?,?,?,?,?)",
                   (_iso(ts), symbol, prob, regime, sentiment, action, reason))

    def log_order(self, ts: datetime, ack) -> None:
        self._exec(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)",
            (_iso(ts), ack.client_order_id, ack.symbol, ack.side, ack.qty, ack.status, ack.filled_price,
             ack.take_profit, ack.stop_loss, json.dumps(ack.extra, default=str)),
        )

    def log_fill(self, ts: datetime, symbol: str, side: str, qty: float, price: float, reason: str,
                 pnl: float | None = None) -> None:
        self._exec("INSERT INTO fills VALUES (?,?,?,?,?,?,?)", (_iso(ts), symbol, side, qty, price, reason, pnl))

    def log_equity(self, ts: datetime, equity: float, cash: float, gross: float, n_pos: int) -> None:
        self._exec("INSERT INTO equity VALUES (?,?,?,?,?)", (_iso(ts), equity, cash, gross, n_pos))

    def log_risk(self, ts: datetime, level: str, rule: str, message: str) -> None:
        self._exec("INSERT INTO risk_events VALUES (?,?,?,?)", (_iso(ts), level, rule, message))

    def read_table(self, table: str, limit: int = 1000) -> pd.DataFrame:
        if table not in {"signals", "orders", "fills", "equity", "risk_events", "news", "sentiment"}:
            raise ValueError(table)
        with self._lock:
            return pd.read_sql_query(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?", self._conn,
                                     params=[limit])
