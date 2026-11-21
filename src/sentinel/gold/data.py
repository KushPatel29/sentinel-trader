"""Daily gold data and the context series that drive it.

Reference price: COMEX gold futures (GC=F). Spot XAU/USD trades a few dollars away (the futures
basis), so the app can rescale every level to the user's own broker quote."""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache

import pandas as pd

from sentinel.analyzer.data import UA, _clean

GOLD = "GC=F"
CONTEXT = {
    "dollar": "DX-Y.NYB",     # US dollar index
    "yield10": "^TNX",        # 10-year Treasury yield (x10 on Yahoo before 2024? we only use differences)
    "tips": "TIP",            # inflation-protected bonds: rise when real yields fall
    "treasuries": "IEF",      # 7-10y nominal Treasuries
    "vix": "^VIX",            # equity fear gauge
    "gvz": "^GVZ",            # gold's own implied volatility (the "gold VIX")
    "stocks": "SPY",
    "silver": "SI=F",
    "copper": "HG=F",
    "oil": "CL=F",
    "gld": "GLD",             # largest gold ETF: its volume is a demand/flow proxy
    "miners": "GDX",          # gold miners
}
START = "2000-01-01"


def _chart_daily(ticker: str, start: str = START) -> pd.DataFrame:
    import requests

    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"interval": "1d", "period1": p1, "period2": int(time.time()), "events": "div,split"},
                     headers=UA, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                       "adj_close": adj, "volume": q["volume"]},
                      index=pd.to_datetime(res["timestamp"], unit="s"))
    return _clean(df.fillna({"volume": 0}))


@lru_cache(maxsize=2)
def _offline(path: str) -> dict:
    return json.load(open(path))


def _offline_daily(ticker: str) -> pd.DataFrame:
    raw = _offline(os.environ["SENTINEL_OFFLINE_GOLD"])["daily"]
    rows = [line.split(",") for line in raw[ticker].split("\n")]
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "adj_close", "volume"]).astype(float)
    df.index = pd.to_datetime(df.pop("t"), unit="s")
    return _clean(df)


def fetch_daily(ticker: str) -> pd.DataFrame:
    if os.environ.get("SENTINEL_OFFLINE_GOLD"):
        return _offline_daily(ticker)
    if os.environ.get("SENTINEL_DISABLE_LIVE"):
        raise RuntimeError("live data disabled (SENTINEL_DISABLE_LIVE)")
    return _chart_daily(ticker)


SESSION_CLOSE_HOUR_NY = 17   # COMEX gold's daily session ends 17:00 New York time


def split_provisional(gold: pd.DataFrame, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Yahoo includes the still-trading day as the last daily bar. Forecasts, ranges and the trade plan must use
    completed sessions only, so an in-progress bar is removed and reported separately as the live quote."""
    now_ny = (now or pd.Timestamp.now(tz="UTC")).tz_convert("America/New_York")
    last_day = gold.index[-1].date()
    today = now_ny.date()
    provisional = last_day > today or (last_day == today and now_ny.hour < SESSION_CLOSE_HOUR_NY)
    live = {"price": float(gold["close"].iloc[-1]), "bar_date": gold.index[-1], "provisional": bool(provisional),
            "as_of": now_ny}
    if provisional and len(gold) > 1:
        gold = gold.iloc[:-1]
    return gold, live


def load_universe(fetch=fetch_daily, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (gold OHLCV of COMPLETED sessions, context closes aligned to gold's calendar plus GLD volume).
    The live quote (possibly from an unfinished session) is in gold.attrs['live']."""
    gold = fetch(GOLD)
    gold = gold[gold["close"] > 0]
    gold, live = split_provisional(gold, now)
    ctx = pd.DataFrame(index=gold.index)
    for key, sym in CONTEXT.items():
        try:
            d = fetch(sym)
        except Exception:
            continue
        ctx[key] = d["close"].reindex(gold.index.union(d.index)).ffill().reindex(gold.index)
        if key == "gld":
            ctx["gld_volume"] = d["volume"].reindex(gold.index)
    gold.attrs["live"] = live
    return gold, ctx
