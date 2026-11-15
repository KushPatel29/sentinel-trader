"""Daily market data, context series, fundamentals and headlines from Yahoo Finance.

Two fetch paths: the `yfinance` library, and Yahoo's public chart endpoint as a fallback (cloud
hosts are sometimes rate-limited by yfinance's cookie flow). Everything is cached by the caller."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

log = logging.getLogger(__name__)

# Context series used as "macro" and "psychology" inputs
CONTEXT = {
    "market": "SPY",        # US equity market
    "vix": "^VIX",          # implied volatility: the market's fear gauge
    "rates": "^TNX",        # 10-year Treasury yield
    "dollar": "DX-Y.NYB",   # US dollar index
}
CRYPTO_MARKET = "BTC-USD"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0 Safari/537.36"}


def asset_class(ticker: str, quote_type: str | None = None) -> str:
    qt = (quote_type or "").upper()
    t = ticker.upper()
    if qt == "CRYPTOCURRENCY" or (t.endswith("-USD") and qt in ("", "CRYPTOCURRENCY")):
        return "crypto"
    if qt == "FUTURE" or t.endswith("=F"):
        return "commodity"
    if qt == "ETF":
        return "etf"
    if qt == "INDEX" or t.startswith("^"):
        return "index"
    if t.endswith("=X") or qt == "CURRENCY":
        return "fx"
    return "stock"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df = df[["open", "high", "low", "close", "adj_close", "volume"]].astype(float)
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    # Back-adjust OHLC with the adjusted close so splits/dividends do not create fake gaps
    f = (df["adj_close"] / df["close"]).fillna(1.0)
    for c in ("open", "high", "low"):
        df[c] = df[c] * f
    df["close"] = df["adj_close"]
    return df.drop(columns="adj_close")


def _chart_api(ticker: str, rng: str = "10y") -> pd.DataFrame:
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    r = requests.get(url, params={"interval": "1d", "range": rng, "events": "div,split"}, headers=UA, timeout=20)
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
    """SENTINEL_OFFLINE_DAILY=<json> serves saved Yahoo data (tests, CI, air-gapped demos)."""
    raw = _offline(os.environ["SENTINEL_OFFLINE_DAILY"])
    if ticker not in raw:
        raise ValueError(f"{ticker} is not in the offline dataset")
    rows = [line.split(",") for line in raw[ticker].split("\n")]
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "adj_close", "volume"]).astype(float)
    df.index = pd.to_datetime(df.pop("t"), unit="s")
    return _clean(df)


def fetch_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    if os.environ.get("SENTINEL_OFFLINE_DAILY"):
        return _offline_daily(ticker)
    try:  # one fast HTTP request; yfinance is the fallback (its cookie flow is often rate-limited on cloud hosts)
        return _chart_api(ticker, period)
    except Exception as e:
        log.warning("chart API failed for %s (%s); trying yfinance", ticker, e)
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False, actions=False, timeout=20)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}. Check the symbol (e.g. AAPL, BTC-USD, GC=F, RELIANCE.NS).")
    return _clean(df)


@dataclass
class Snapshot:
    """Point-in-time information with no usable history (shown and scored, never backtested)."""
    name: str = ""
    quote_type: str = ""
    currency: str = ""
    info: dict = field(default_factory=dict)
    headlines: list[dict] = field(default_factory=list)


def fetch_snapshot(ticker: str) -> Snapshot:
    snap = Snapshot()
    if os.environ.get("SENTINEL_OFFLINE_DAILY"):
        t = ticker.upper()
        snap.name = t
        snap.quote_type = ("CRYPTOCURRENCY" if t.endswith("-USD") else "FUTURE" if t.endswith("=F")
                           else "ETF" if t in ("SPY", "QQQ", "TLT") else "EQUITY")
        return snap
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}
        snap.info = info
        snap.name = info.get("longName") or info.get("shortName") or ticker
        snap.quote_type = info.get("quoteType", "")
        snap.currency = info.get("currency", "")
        try:
            for n in (t.news or [])[:15]:
                c = n.get("content", n)
                title = c.get("title") or n.get("title")
                if not title:
                    continue
                prov = c.get("provider") or {}
                snap.headlines.append({
                    "title": title, "summary": c.get("summary", "") or "",
                    "publisher": prov.get("displayName", "") if isinstance(prov, dict) else n.get("publisher", ""),
                    "time": c.get("pubDate") or n.get("providerPublishTime"),
                    "link": (c.get("canonicalUrl") or {}).get("url", "") if isinstance(c.get("canonicalUrl"), dict)
                    else n.get("link", ""),
                })
        except Exception as e:
            log.warning("news unavailable for %s: %s", ticker, e)
    except ImportError:
        pass
    return snap


ALIASES = {
    "gold": ("GC=F", "Gold futures"), "silver": ("SI=F", "Silver futures"), "oil": ("CL=F", "Crude oil (WTI) futures"),
    "crude": ("CL=F", "Crude oil (WTI) futures"), "brent": ("BZ=F", "Brent crude futures"),
    "natural gas": ("NG=F", "Natural gas futures"), "gas": ("NG=F", "Natural gas futures"),
    "copper": ("HG=F", "Copper futures"), "platinum": ("PL=F", "Platinum futures"), "corn": ("ZC=F", "Corn futures"),
    "wheat": ("ZW=F", "Wheat futures"), "soybean": ("ZS=F", "Soybean futures"), "coffee": ("KC=F", "Coffee futures"),
    "bitcoin": ("BTC-USD", "Bitcoin USD"), "btc": ("BTC-USD", "Bitcoin USD"), "ethereum": ("ETH-USD", "Ethereum USD"),
    "eth": ("ETH-USD", "Ethereum USD"), "solana": ("SOL-USD", "Solana USD"), "xrp": ("XRP-USD", "XRP USD"),
    "dogecoin": ("DOGE-USD", "Dogecoin USD"), "nifty": ("^NSEI", "NIFTY 50"), "sensex": ("^BSESN", "S&P BSE SENSEX"),
    "s&p 500": ("^GSPC", "S&P 500 index"), "nasdaq": ("^IXIC", "Nasdaq Composite"),
}


def search(query: str, limit: int = 8) -> list[dict]:
    """Name/ticker search across stocks, ETFs, crypto, futures. Common commodity/crypto words map
    straight to their futures/crypto tickers first (Yahoo's search would return e.g. a stock named 'Gold')."""
    hit = ALIASES.get(query.strip().lower())
    alias = [{"symbol": hit[0], "name": hit[1], "type": "", "exchange": ""}] if hit else []
    rest = _search(query, limit)
    return alias + [r for r in rest if not alias or r["symbol"] != alias[0]["symbol"]]


def _search(query: str, limit: int) -> list[dict]:
    if os.environ.get("SENTINEL_OFFLINE_DAILY"):
        raw = _offline(os.environ["SENTINEL_OFFLINE_DAILY"])
        return [{"symbol": k, "name": k, "type": "", "exchange": "offline"} for k in raw if query.upper() in k][:limit]
    try:
        import yfinance as yf

        quotes = yf.Search(query, max_results=limit, news_count=0).quotes
    except Exception:
        try:
            import requests

            r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                             params={"q": query, "quotesCount": limit, "newsCount": 0}, headers=UA, timeout=10)
            quotes = r.json().get("quotes", [])
        except Exception:
            return []
    return [{"symbol": q.get("symbol"), "name": q.get("shortname") or q.get("longname") or "",
             "type": q.get("quoteType", ""), "exchange": q.get("exchDisp") or q.get("exchange", "")}
            for q in quotes if q.get("symbol")]


def fetch_context(asset: str, fetch=fetch_history) -> dict[str, pd.DataFrame]:
    ctx = {}
    for key, sym in CONTEXT.items():
        try:
            ctx[key] = fetch(sym)
        except Exception as e:
            log.warning("context %s unavailable: %s", sym, e)
    if asset == "crypto":
        try:
            ctx["crypto_market"] = fetch(CRYPTO_MARKET)
        except Exception:
            pass
    return ctx


def fetch_intraday(symbols: list[str], minutes: int = 15, period: str = "60d") -> pd.DataFrame:
    """Recent intraday bars (Yahoo keeps ~60 days of 15-minute data) in Sentinel's bar format."""
    from sentinel.market_time import filter_regular_session

    if os.environ.get("SENTINEL_OFFLINE_INTRADAY"):
        raw = _offline(os.environ["SENTINEL_OFFLINE_INTRADAY"])
        rows = [(s, *line.split(",")) for s in symbols if s in raw for line in raw[s].split("\n")]
        df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    else:
        frames = []
        for s in symbols:
            try:
                import yfinance as yf

                d = yf.Ticker(s).history(period=period, interval=f"{minutes}m", prepost=False, auto_adjust=False)
                d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                d.index = pd.to_datetime(d.index).tz_convert("UTC")
            except Exception:
                import requests

                r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s}",
                                 params={"interval": f"{minutes}m", "range": period}, headers=UA, timeout=20)
                res = r.json()["chart"]["result"][0]
                q = res["indicators"]["quote"][0]
                d = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                                 index=pd.to_datetime(res["timestamp"], unit="s", utc=True))
            d = d.dropna(subset=["close"]).assign(symbol=s)
            frames.append(d.rename_axis("ts").reset_index())
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df = filter_regular_session(df[["symbol", "ts", "open", "high", "low", "close", "volume"]], minutes)
    return df.drop_duplicates(["symbol", "ts"]).sort_values(["ts", "symbol"]).reset_index(drop=True)
