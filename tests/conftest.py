from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sentinel.config import Config
from sentinel.data.synthetic import generate
from sentinel.events import Bar
from sentinel.market_time import NY


@pytest.fixture
def cfg() -> Config:
    c = Config(symbols=["AAA", "BBB", "CCC"], timeframe_minutes=15)
    c.model.min_train_bars = 600
    c.model.test_bars = 300
    c.model.lgbm = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 30,
                    "verbose": -1}
    return c


@pytest.fixture(scope="session")
def synth():
    return generate(["AAA", "BBB", "CCC"], n_days=60, minutes=15, seed=3)


def ny(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=NY).astimezone(timezone.utc)


def make_bars(symbol: str, start: datetime, rows: list[tuple], minutes: int = 15) -> pd.DataFrame:
    """rows: (open, high, low, close) tuples at consecutive bars from start."""
    data = [(symbol, start + timedelta(minutes=minutes * i), o, h, lo, c, 1000.0)
            for i, (o, h, lo, c) in enumerate(rows)]
    df = pd.DataFrame(data, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def bar(symbol, ts, o, h, lo, c, minutes=15) -> Bar:
    return Bar(symbol, ts, o, h, lo, c, 1000.0, minutes)
