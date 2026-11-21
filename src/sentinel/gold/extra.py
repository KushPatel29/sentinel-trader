"""Additional gold drivers shown as live context: speculator positioning (CFTC), official real yields and
inflation expectations (FRED), policy uncertainty, credit stress, Fed policy, Asian currency demand, Bitcoin.

Each was added to the forecasting models and tested on 2019-2026 data (see research/gold). None improved
direction or range forecasts, so they inform the 'what's moving gold' view only. Every fetch is best-effort:
a source that is down simply drops out of the table."""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd

from sentinel.analyzer.data import UA
from sentinel.gold.data import fetch_daily

YAHOO_EXTRA = {"credit_hy": "HYG", "credit_ig": "LQD", "tbill": "^IRX", "inr": "INR=X", "cny": "CNY=X",
               "btc": "BTC-USD"}
FRED = {"real_yield": "DFII10", "breakeven": "T10YIE", "policy_uncertainty": "USEPUINDXD"}
COT_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"


def fetch_cot() -> pd.DataFrame:
    """Managed-money (hedge fund) and producer positions in COMEX gold, weekly."""
    if os.environ.get("SENTINEL_OFFLINE_COT"):
        rows = json.load(open(os.environ["SENTINEL_OFFLINE_COT"]))
    else:
        if os.environ.get("SENTINEL_DISABLE_LIVE"):
            raise RuntimeError("live data disabled")
        import requests

        rows = requests.get(COT_URL, params={
            "cftc_contract_market_code": "088691",
            "$select": "report_date_as_yyyy_mm_dd,m_money_positions_long_all,m_money_positions_short_all,"
                       "open_interest_all",
            "$order": "report_date_as_yyyy_mm_dd", "$limit": "5000"}, headers=UA, timeout=20).json()
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d.pop("report_date_as_yyyy_mm_dd"))
    d = d.set_index("date").astype(float).sort_index()
    net = (d["m_money_positions_long_all"] - d["m_money_positions_short_all"]) / d["open_interest_all"]
    return pd.DataFrame({"mm_net_pct_oi": net, "mm_long": d["m_money_positions_long_all"],
                         "mm_short": d["m_money_positions_short_all"]})


def fetch_fred(series_id: str) -> pd.Series:
    if os.environ.get("SENTINEL_DISABLE_LIVE") or os.environ.get("SENTINEL_OFFLINE_GOLD"):
        raise RuntimeError("live data disabled")
    import requests

    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id}, headers=UA,
                     timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def _extra_daily(sym: str) -> pd.Series:
    path = os.environ.get("SENTINEL_OFFLINE_GOLD_EXTRA")
    if path:
        raw = json.load(open(path))["daily"]
        if sym not in raw:
            raise KeyError(sym)
        rows = [line.split(",") for line in raw[sym].split("\n")]
        df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "a", "v"]).astype(float)
        return pd.Series(df["c"].values, index=pd.to_datetime(df["t"], unit="s").dt.normalize())
    return fetch_daily(sym)["close"]


def extended_drivers() -> pd.DataFrame:
    """One row per extra driver: latest value, 1-month change, where it sits in its 3-year range."""
    rows = []

    def add(name, value, change, pct, unit, why):
        rows.append({"driver": name, "latest": value, "1-month change": change, "3-year percentile": pct,
                     "unit": unit, "why it matters": why})

    def pct_rank(s: pd.Series) -> float:
        s = s.dropna().iloc[-756:]
        return float((s <= s.iloc[-1]).mean()) if len(s) else np.nan

    try:
        cot = fetch_cot()["mm_net_pct_oi"]
        add("Hedge-fund net position (CFTC)", cot.iloc[-1], cot.iloc[-1] - cot.iloc[-5], pct_rank(cot), "% of open interest",
            "Crowded long positioning can unwind fast; washed-out positioning removes sellers")
    except Exception:
        pass
    for _key, sid, name, unit, why in (
        ("real_yield", "DFII10", "10-year real yield (TIPS)", "%", "Gold's classic opposite: higher real yields make "
         "interest-free gold less attractive"),
        ("breakeven", "T10YIE", "10-year inflation expectation", "%", "Expected inflation; gold is a traditional hedge"),
        ("policy_uncertainty", "USEPUINDXD", "US policy uncertainty index", "index", "Uncertainty drives haven demand"),
    ):
        try:
            s = fetch_fred(sid)
            add(name, s.iloc[-1], s.iloc[-1] - s.asof(s.index[-1] - pd.Timedelta(days=30)), pct_rank(s), unit, why)
        except Exception:
            pass
    try:
        hy, ig = _extra_daily("HYG"), _extra_daily("LQD")
        ratio = (hy / ig.reindex(hy.index).ffill()).dropna()
        add("Credit stress (high-yield vs investment-grade)", ratio.iloc[-1], ratio.pct_change(21).iloc[-1],
            pct_rank(ratio), "ratio", "Falling ratio = credit stress; can trigger haven buying or forced selling")
    except Exception:
        pass
    for sym, name, unit, why in (
        ("^IRX", "3-month T-bill yield (Fed policy)", "%", "Higher policy rates raise the cost of holding gold"),
        ("INR=X", "US dollar vs Indian rupee", "INR", "India is a top-2 buyer: a weaker rupee makes gold dearer locally"),
        ("CNY=X", "US dollar vs Chinese yuan", "CNY", "China is the largest buyer; yuan weakness can lift local demand"),
        ("BTC-USD", "Bitcoin", "USD", "Competes with gold as an alternative store of value"),
    ):
        try:
            s = _extra_daily(sym).dropna()
            chg = s.iloc[-1] - s.asof(s.index[-1] - pd.Timedelta(days=30)) if sym == "^IRX" else \
                s.iloc[-1] / s.asof(s.index[-1] - pd.Timedelta(days=30)) - 1
            add(name, s.iloc[-1], chg, pct_rank(s), unit, why)
        except Exception:
            pass
    return pd.DataFrame(rows)
