"""Checksum-verified public Binance spot archives plus incremental public REST data.

No credentials, private endpoints or order placement exist in this module.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ROOT, atomic_json

API = "https://api.binance.com"
ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines"
BAR = pd.Timedelta(hours=4)
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "SentinelCryptoResearch/0.2"
    return s


def valid_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if not re.fullmatch(r"[A-Z0-9]{2,16}USDT", symbol):
        raise ValueError("Expected a spot USDT symbol, e.g. BTCUSDT")
    return symbol


def server_now() -> pd.Timestamp:
    with session() as s:
        r = s.get(API + "/api/v3/time", timeout=20)
        r.raise_for_status()
    return pd.to_datetime(r.json()["serverTime"], unit="ms", utc=True)


def normalize(raw: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    raw = raw.copy()
    ts = pd.to_numeric(raw["timestamp"], errors="raise").astype("int64")
    # Binance archives changed from milliseconds to microseconds in January 2025.
    ts = ts.where(ts < 100_000_000_000_000, ts // 1000)
    raw.index = pd.to_datetime(ts, unit="ms", utc=True)
    raw.index.name = "timestamp"
    cols = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base"]
    out = raw[cols].apply(pd.to_numeric, errors="raise").astype(float)
    out = out.sort_index()[lambda x: ~x.index.duplicated(keep="last")]
    out = out[out.index + BAR <= now]  # Never train on an unfinished candle.
    validate(out)
    return out


def validate(df: pd.DataFrame, require_contiguous: bool = False) -> dict:
    if df.empty:
        raise ValueError("No completed candles")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Nonfinite market data")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise ValueError("Invalid price/volume")
    if ((df.high < df[["open", "close", "low"]].max(axis=1)) |
        (df.low > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC bounds")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Timestamps must be increasing and unique")
    off_grid = ((df.index.hour % 4 != 0) | (df.index.minute != 0) | (df.index.second != 0)).sum()
    if off_grid:
        raise ValueError("Candles are not aligned to the UTC 4h grid")
    gaps = int((df.index.to_series().diff().iloc[1:] != BAR).sum())
    if require_contiguous and gaps:
        raise ValueError(f"{gaps} gaps: refuse a backtest across missing candles")
    return {"bars": len(df), "start": str(df.index[0]), "last_open": str(df.index[-1]), "gaps": gaps}


def _archive(symbol: str, month: str, root: Path) -> dict:
    name = f"{symbol}-4h-{month}.zip"
    url = f"{ARCHIVE}/{symbol}/4h/{name}"
    folder = root / "data" / "raw" / symbol
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    checksum_path = folder / (name + ".CHECKSUM")
    with session() as s:
        if not checksum_path.exists():
            r = s.get(url + ".CHECKSUM", timeout=30)
            r.raise_for_status()
            checksum_path.write_text(r.text, encoding="utf-8")
        expected = checksum_path.read_text().split()[0]
        if not path.exists():
            r = s.get(url, timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Archive checksum mismatch: {path}")
    return {"symbol": symbol, "month": month, "url": url, "sha256": actual, "verified": True}


def rest_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp, s=None) -> pd.DataFrame:
    own = s is None
    s = s or session()
    rows = []
    cursor = int(start.timestamp() * 1000)
    try:
        while cursor < int(end.timestamp() * 1000):
            r = s.get(API + "/api/v3/klines", params={"symbol": valid_symbol(symbol), "interval": "4h",
                      "startTime": cursor, "endTime": int(end.timestamp() * 1000), "limit": 1000}, timeout=25)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            if not isinstance(batch, list):
                raise ValueError(f"Invalid exchange response: {batch}")
            rows.extend(batch)
            new_cursor = int(batch[-1][0]) + 14_400_000
            if new_cursor <= cursor:
                raise ValueError("REST pagination did not advance")
            cursor = new_cursor
            time.sleep(0.12)
    finally:
        if own:
            s.close()
    if not rows:
        return pd.DataFrame()
    return normalize(pd.DataFrame(rows, columns=COLUMNS), end)


def fetch_history(cfg: dict, root: Path = ROOT) -> dict[str, pd.DataFrame]:
    now = server_now()
    cutoff = pd.Timestamp(cfg["data_cutoff"])
    if cutoff > now:
        raise ValueError("Protocol cutoff is in the future")
    first = pd.Timestamp(cfg["start"], tz="UTC")
    current_month = cutoff.normalize().replace(day=1)
    months = pd.date_range(first, current_month, freq="MS", inclusive="left").strftime("%Y-%m")
    tasks = [(valid_symbol(sym), month) for sym in cfg["symbols"] for month in months]
    manifest = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_archive, sym, month, root) for sym, month in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            manifest.append(f.result())
            if i % 30 == 0:
                print(f"Verified {i}/{len(tasks)} archives", flush=True)
    out, quality = {}, {}
    for symbol in cfg["symbols"]:
        chunks = []
        for month in months:
            path = root / "data" / "raw" / symbol / f"{symbol}-4h-{month}.zip"
            with zipfile.ZipFile(path) as z:
                chunks.append(normalize(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, names=COLUMNS), cutoff))
        recent = rest_bars(symbol, current_month, cutoff)
        if not recent.empty:
            chunks.append(recent)
        df = pd.concat(chunks).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df[(df.index >= first) & (df.index + BAR <= cutoff)]
        quality[symbol] = validate(df, require_contiguous=True)
        path = root / "data" / f"{symbol}-4h.csv"
        df.to_csv(path)
        quality[symbol]["csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        out[symbol] = df
    atomic_json(root / "data" / "manifest.json", {"retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "exchange_utc": str(now), "quality": quality, "archives": sorted(manifest, key=lambda m: (m["symbol"], m["month"]))})
    return out


def read_history(symbol: str, root: Path = ROOT) -> pd.DataFrame:
    df = pd.read_csv(root / "data" / f"{valid_symbol(symbol)}-4h.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    validate(df, require_contiguous=True)
    return df


def live_snapshot(symbol: str, s=None) -> dict:
    own = s is None
    s = s or session()
    try:
        sent = time.monotonic()
        r = s.get(API + "/api/v3/ticker/bookTicker", params={"symbol": valid_symbol(symbol)}, timeout=15)
        r.raise_for_status()
        book = r.json()
        bid, ask = float(book["bidPrice"]), float(book["askPrice"])
        if bid <= 0 or ask < bid:
            raise ValueError("Invalid quote")
        return {"symbol": symbol, "bid": bid, "ask": ask, "mid": (bid + ask) / 2,
                "spread_bps": (ask / bid - 1) * 10000,
                "received_utc": datetime.now(timezone.utc).isoformat(), "latency_s": time.monotonic() - sent}
    finally:
        if own:
            s.close()
