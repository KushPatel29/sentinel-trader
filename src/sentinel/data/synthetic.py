"""Synthetic market + news generator for offline demos and tests.

Prices follow a regime-switching random walk with intraday volatility seasonality. A fraction of
headlines are "informative": they precede a drift in the direction of their wording, so the
sentiment pipeline has something real to find. Results on this data say nothing about real markets;
it exists so the whole pipeline can be run and tested without API keys."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from sentinel.events import NewsItem
from sentinel.market_time import NY

POS = ["beats earnings estimates", "raises full-year guidance", "wins major contract", "upgraded to buy",
       "reports record revenue", "announces buyback", "surges on strong demand"]
NEG = ["misses earnings estimates", "cuts guidance", "faces SEC investigation", "downgraded to sell",
       "announces product recall", "hit with lawsuit", "plunges after weak outlook"]
NEUTRAL = ["to present at investor conference", "names new board member", "schedules earnings call",
           "files quarterly report", "opens new office"]


def trading_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def generate(symbols: list[str], n_days: int = 250, minutes: int = 15, seed: int = 7,
             start: date = date(2024, 1, 2)) -> tuple[pd.DataFrame, list[NewsItem]]:
    rng = np.random.default_rng(seed)
    days = trading_days(start, n_days)
    bars_per_day = 390 // minutes
    rows, news = [], []
    for si, sym in enumerate(symbols):
        price = 50 + 150 * rng.random()
        regime = 0
        drift_left, drift = 0, 0.0
        for d in days:
            open_ny = datetime(d.year, d.month, d.day, 9, 30, tzinfo=NY)
            if rng.random() < 0.05:
                regime = int(rng.integers(0, 3))
            base_vol = [0.0015, 0.003, 0.006][regime] * np.sqrt(minutes / 15)
            # overnight gap
            price *= float(np.exp(rng.normal(0, base_vol * 2)))
            for b in range(bars_per_day):
                ts = (open_ny + timedelta(minutes=b * minutes)).astimezone(timezone.utc)
                # news arrives mid-session with ~0.6 probability per symbol-day
                if b == int(bars_per_day * rng.random()) and rng.random() < 0.6:
                    kind = rng.choice(["pos", "neg", "neu"], p=[0.35, 0.35, 0.30])
                    informative = rng.random() < 0.7
                    text = {"pos": POS, "neg": NEG, "neu": NEUTRAL}[kind]
                    headline = f"{sym} {rng.choice(text)}"
                    news.append(NewsItem(id=f"syn-{sym}-{d.isoformat()}-{b}", ts=ts + timedelta(minutes=1),
                                         headline=headline, symbols=(sym,), source="synthetic"))
                    if informative and kind != "neu":
                        drift_left = int(rng.integers(3, 10))
                        drift = (1 if kind == "pos" else -1) * base_vol * 0.8
                # momentum-ish microstructure: weak autocorrelation of returns
                tod = b / bars_per_day
                seasonal = 1.4 if tod < 0.1 or tod > 0.9 else 1.0
                mu = drift if drift_left > 0 else 0.0
                drift_left = max(0, drift_left - 1)
                r = rng.normal(mu, base_vol * seasonal)
                o = price
                c = o * float(np.exp(r))
                spread = abs(rng.normal(0, base_vol * seasonal * 0.6))
                h = max(o, c) * (1 + spread)
                lo = min(o, c) * (1 - spread)
                vol = float(rng.lognormal(10 + si * 0.1, 0.4) * seasonal * (1 + 5 * abs(r) / base_vol / 10))
                rows.append((sym, ts, o, h, lo, c, vol))
                price = c
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    news.sort(key=lambda n: n.ts)
    return df.sort_values(["ts", "symbol"]).reset_index(drop=True), news
