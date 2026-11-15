"""Public reference CFD bars, public market context, and broker CSV ingestion.

HistData M1 quotes are bid prices in fixed EST (UTC-5), without DST and without
aggregated volume. No zero volume is promoted to a demand/order-flow feature.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import time
from urllib.parse import quote
import zipfile

import numpy as np
import pandas as pd
import requests

from .config import ROOT, atomic_json
from .data import read_history, session


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form" and a.get("id") == "file_down":
            self.active = True
        if self.active and tag == "input" and a.get("name"):
            self.fields[a["name"]] = a.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def load_focus(root: Path = ROOT):
    return json.loads((root / "focus_protocol.json").read_text())


def reference_archive(symbol: str, year: int, month: int | None, root: Path = ROOT):
    if symbol not in ("XAUUSD", "WTIUSD"):
        raise ValueError("Unsupported reference CFD symbol")
    label = str(year) + (f"-{month:02d}" if month else "")
    path = root / "data" / "focus" / "raw" / symbol / f"{label}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{symbol.lower()}/{year}"
    if month:
        url += f"/{month}"
    if not path.exists():
        with session() as s:
            r = s.get(url, timeout=(6, 30))
            r.raise_for_status()
            form = DownloadForm()
            form.feed(r.text)
            if not form.fields.get("tk"):
                return None
            r = s.post("https://www.histdata.com/get.php", data=form.fields, headers={"Referer": url}, timeout=(6, 90))
            r.raise_for_status()
            if not zipfile.is_zipfile(io.BytesIO(r.content)):
                raise ValueError(f"Expected ZIP from {url}")
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                if z.testzip() is not None:
                    raise ValueError("Corrupt reference archive")
            temp = path.with_suffix(".tmp")
            temp.write_bytes(r.content)
            temp.replace(path)
        time.sleep(.2)
    return {"symbol": symbol, "period": label, "url": url, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "integrity": "ZIP CRC verified; locally recorded SHA-256, no provider checksum supplied"}


def parse_reference(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    chunks = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            for raw in pd.read_csv(z.open(name), sep=";", header=None,
                                   names=["time", "open", "high", "low", "close", "unused_volume"],
                                   chunksize=250000):
                # Fixed EST: do not localize to America/New_York (which applies DST).
                idx = pd.to_datetime(raw.pop("time"), format="%Y%m%d %H%M%S", errors="raise")
                raw.index = pd.DatetimeIndex(idx).tz_localize(timezone(pd.Timedelta(hours=-5))).tz_convert("UTC")
                raw = raw.drop(columns="unused_volume")
                raw = raw[raw.index + pd.Timedelta(minutes=1) <= cutoff]
                if raw.empty:
                    continue
                agg = raw.resample("1h", label="left", closed="left").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last"})
                agg["minute_count"] = raw.close.resample("1h").count()
                chunks.append(agg.dropna(subset=["close"]))
    if not chunks:
        raise ValueError(f"No bars in {path}")
    # Chunk edges can split an hour. Aggregate their already-causal partial OHLCs.
    out = pd.concat(chunks).sort_index()
    out = out.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last", "minute_count": "sum"})
    return out[out.index + pd.Timedelta(hours=1) <= cutoff]


def check_reference(df: pd.DataFrame) -> dict:
    required = ["open", "high", "low", "close"]
    if df.empty or not np.isfinite(df[required].to_numpy()).all():
        raise ValueError("Empty/nonfinite reference prices")
    if (df[required] <= 0).any().any():
        raise ValueError("Nonpositive prices: a CFD return model cannot assume lognormal prices here")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered timestamps")
    if ((df.high < df[required].max(axis=1)) | (df.low > df[required].min(axis=1))).any():
        raise ValueError("Invalid OHLC range")
    gap = df.index.to_series().diff().dt.total_seconds().div(3600)
    return {"bars": len(df), "start": str(df.index[0]), "last_open": str(df.index[-1]),
            "max_gap_hours": float(gap.max()), "gaps_over_4h": int((gap > 4).sum()),
            "unexplained_midweek_gaps_over_4h": int(((gap > 4) & (df.index.dayofweek.isin([1, 2, 3, 4]))).sum()),
            "partial_hour_bars": int((df.get("minute_count", pd.Series(60, index=df.index)) < 55).sum()),
            "volume": "unavailable; not used", "price_side": "bid; assumed spread added to execution"}


def yahoo_context(symbol: str, start: str, cutoff: str, root: Path = ROOT):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    params = {"period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
              "period2": int(pd.Timestamp(cutoff).timestamp()), "interval": "1d", "events": "div,splits"}
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=(6, 30))
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True).normalize()
    df = pd.DataFrame({"close": adj}, index=idx).dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    path = root / "data" / "focus" / "context" / (symbol.replace("^", "") + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df, {"symbol": symbol, "url": r.url, "bars": len(df),
                "sha256": hashlib.sha256(r.content).hexdigest(), "type": "adjusted daily context, not CFD execution data"}


def fetch_focus(root: Path = ROOT) -> dict:
    cfg = load_focus(root)
    cutoff = pd.Timestamp(cfg["cutoff"])
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(), "archives": [], "quality": {}, "context": {}}
    # Two concurrent downloads respect the free site's resources. Completed years first.
    annual = [(sym, year, None) for sym in ("XAUUSD", "WTIUSD") for year in range(int(cfg["start"][:4]), cutoff.year)]
    missing = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(reference_archive, sym, year, month, root): (sym, year) for sym, year, month in annual}
        for fut in as_completed(tasks):
            result = fut.result()
            if result:
                manifest["archives"].append(result)
                print(f"Downloaded {result['symbol']} {result['period']}", flush=True)
            else:
                missing.append(tasks[fut])
    monthly = [(sym, year, month) for sym, year in missing for month in range(1, 13)]
    monthly += [(sym, cutoff.year, month) for sym in ("XAUUSD", "WTIUSD") for month in range(1, cutoff.month + 1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(reference_archive, *args, root): args for args in monthly}
        for i, fut in enumerate(as_completed(tasks), 1):
            result = fut.result()
            if result:
                manifest["archives"].append(result)
            else:
                sym, year, month = tasks[fut]
                manifest.setdefault("missing_periods", []).append({"symbol": sym, "year": year, "month": month})
            if i % 15 == 0:
                print(f"Processed {i}/{len(monthly)} monthly requests", flush=True)
    for sym in ("XAUUSD", "WTIUSD"):
        parts = []
        for item in sorted([r for r in manifest["archives"] if r["symbol"] == sym], key=lambda r: r["period"]):
            parts.append(parse_reference(Path(item["path"]), cutoff))
        if not parts:
            raise ValueError(f"No real reference data for {sym}")
        df = pd.concat(parts).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        manifest["quality"][sym] = check_reference(df)
        # No interpolation across closed sessions or missing data.
        df.to_csv(root / "data" / "focus" / f"{sym}.csv", index_label="timestamp")
        print(f"Prepared {sym}: {len(df)} hourly reference bars", flush=True)
    for sym in cfg["assets"]:
        if sym.endswith("USDT"):
            df = read_history(sym, root)
            df = df[df.index + pd.Timedelta(hours=4) <= cutoff]
            df.to_csv(root / "data" / "focus" / f"{sym}.csv", index_label="timestamp")
            manifest["quality"][sym] = {"bars": len(df), "start": str(df.index[0]), "last_open": str(df.index[-1]),
                                        "type": "Binance verified spot archive, 4h"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = {pool.submit(yahoo_context, sym, cfg["start"], cfg["cutoff"], root): key for key, sym in cfg["context_symbols"].items()}
        for fut in as_completed(tasks):
            _, metadata = fut.result()
            manifest["context"][tasks[fut]] = metadata
    atomic_json(root / "data" / "focus" / "manifest.json", manifest)
    return manifest


def read_focus(symbol, root: Path = ROOT):
    df = pd.read_csv(root / "data" / "focus" / f"{symbol}.csv", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def fetch_oil_reference(root: Path = ROOT):
    """Keep a continuous futures direction study separate from the incomplete CFD feed."""
    cfg = load_focus(root)
    url = "https://query1.finance.yahoo.com/v8/finance/chart/CL%3DF"
    r = requests.get(url, params={"period1": 1609459200, "period2": int(pd.Timestamp(cfg["cutoff"]).timestamp()),
                                 "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=(6, 30))
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(q, index=pd.to_datetime(result["timestamp"], unit="s", utc=True).normalize())
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    check_reference(df)
    df.to_csv(root / "data" / "focus" / "WTI_REFERENCE.csv", index_label="timestamp")
    meta = {"bars": len(df), "start": str(df.index[0]), "last_open": str(df.index[-1]), "url": r.url,
            "sha256": hashlib.sha256(r.content).hexdigest(), "type": "continuous front-month futures, direction only",
            "warning": "Contract-roll price differences are not cash P&L. Not USOIL CFD execution data; 2020 negative-price period not included."}
    manifest_path = root / "data" / "focus" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["quality"]["WTI_REFERENCE"] = meta
    atomic_json(manifest_path, manifest)
    return meta


def import_broker_csv(path: Path, symbol: str, root: Path = ROOT):
    """Explicit broker timestamp/side schema; never silently rename a futures/ETF price."""
    raw = pd.read_csv(path)
    required = ["timestamp", "bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_close"]
    if any(c not in raw for c in required):
        raise ValueError("Broker CSV requires explicit bid/ask columns: " + ", ".join(required))
    text = raw.timestamp.astype(str)
    if not text.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all():
        raise ValueError("Every broker timestamp must include an explicit UTC offset")
    raw.index = pd.to_datetime(raw.pop("timestamp"), utc=True)
    raw = raw.sort_index()
    df = raw.rename(columns={"bid_" + c: c for c in ("open", "high", "low", "close")})
    check_reference(df)
    if ((df.ask_open < df.open) | (df.ask_close < df.close)).any():
        raise ValueError("Ask below bid")
    if symbol not in ("XAUUSD", "WTIUSD", "USOIL"):
        raise ValueError("Expected XAUUSD or USOIL/WTIUSD")
    target = root / "data" / "broker" / f"{symbol}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index_label="timestamp")
    atomic_json(target.with_suffix(".metadata.json"), {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "symbol": symbol, "imported_utc": datetime.now(timezone.utc).isoformat(), "bars": len(df),
                "execution_ready": False, "required_next": "Verify broker contract size, sessions, fees and financing before replay."})
    return target
