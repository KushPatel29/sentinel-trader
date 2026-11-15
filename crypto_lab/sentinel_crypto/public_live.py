"""Anonymous public market data and hash-verified research-model inference.

Only GET requests to Binance's documented market-data-only host are allowed.
No account keys, authenticated endpoints, order methods or local ledgers are used.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from .accuracy import release_probability
from .config import ROOT
from .data import COLUMNS, normalize, valid_symbol

PUBLIC_API="https://data-api.binance.vision"
SYMBOLS=("BTCUSDT","ETHUSDT","SOLUSDT")


def get_json(path,params=None):
    if os.environ.get("SENTINEL_DISABLE_LIVE")=="1":
        raise RuntimeError("Live market requests are disabled in this environment")
    allowed={"/api/v3/time","/api/v3/ticker/bookTicker","/api/v3/klines"}
    if path not in allowed:
        raise ValueError("Only public market-data endpoints are supported")
    r=requests.get(PUBLIC_API+path,params=params,timeout=(5,15))
    r.raise_for_status()
    return r.json()


def quote_snapshot():
    now=pd.to_datetime(get_json("/api/v3/time")["serverTime"],unit="ms",utc=True)
    books=get_json("/api/v3/ticker/bookTicker",{"symbols":json.dumps(list(SYMBOLS),separators=(",",":"))})
    result=[]
    for row in books:
        bid,ask=float(row["bidPrice"]),float(row["askPrice"])
        if bid<=0 or ask<bid or not np.isfinite([bid,ask]).all():
            raise ValueError("Invalid market quote")
        result.append({"symbol":row["symbol"],"bid":bid,"ask":ask,"mid":(bid+ask)/2,
                       "spread_bps":(ask/bid-1)*10000,"observed_utc":str(now),"venue":"Binance spot · USDT"})
    if {row["symbol"] for row in result}!=set(SYMBOLS) or len(result)!=len(SYMBOLS):
        raise ValueError("Market quote response did not match the supported assets")
    return sorted(result,key=lambda row:SYMBOLS.index(row["symbol"]))


def recent_bars(symbol):
    symbol=valid_symbol(symbol)
    if symbol not in SYMBOLS:
        raise ValueError("This release supports BTC, ETH and SOL")
    now=pd.to_datetime(get_json("/api/v3/time")["serverTime"],unit="ms",utc=True)
    latest=get_json("/api/v3/klines",{"symbol":symbol,"interval":"4h","limit":1000})
    if not latest:
        raise ValueError("No completed price history")
    earlier=get_json("/api/v3/klines",{"symbol":symbol,"interval":"4h","limit":600,"endTime":int(latest[0][0])-1})
    frame=normalize(pd.DataFrame(earlier+latest,columns=COLUMNS),now)
    if len(frame)<1200:
        raise ValueError("Insufficient history for model feature parity")
    gaps=frame.index.to_series().diff().iloc[1:]
    if (gaps!=pd.Timedelta(hours=4)).any():
        raise ValueError("Market history contains missing candles")
    return frame,now


def load_release(symbol,root:Path=ROOT):
    if symbol not in SYMBOLS:
        raise ValueError("CFD live inference is disabled until broker data is qualified")
    folder=root/"release_models"
    manifest=json.loads((folder/"manifest.json").read_text())
    entry=manifest["assets"][symbol]
    # The path is pinned to a known symbol, not controlled by a remote manifest value.
    path=folder/f"{symbol}.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
        raise ValueError("Research artifact integrity check failed")
    return joblib.load(path)


def inference(symbol,bundle,bars,now):
    now=pd.Timestamp(now)
    latest_close=bars.index[-1]+pd.Timedelta(hours=4)
    model_age=(now-pd.Timestamp(bundle["fit_before"])).total_seconds()/86400
    if now-latest_close>pd.Timedelta(hours=4,minutes=5):
        raise ValueError("Price candles are stale")
    if model_age>45:
        raise ValueError("Model artifact is over 45 days old; refresh and validate it before inference")
    p=release_probability(bundle,bars)
    if not np.isfinite(p) or not 0<=p<=1:
        raise ValueError("Invalid model probability")
    return {"symbol":symbol,"up_probability":p,"horizon":"next six 4h bars (24h research target)",
            "last_completed_candle_close":str(latest_close),"model_fit_before":bundle["fit_before"],
            "profile":bundle["profile"],"model_age_days":model_age,"observed_utc":str(now),
            "trade_status":"NO QUALIFIED TRADE · research probability only","orders_enabled":False}
