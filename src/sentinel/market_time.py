"""NYSE regular-session helpers. Holidays are handled by the broker clock in live mode;
offline, data simply does not exist on holidays so the simple session model is sufficient."""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
OPEN = time(9, 30)
CLOSE = time(16, 0)


def to_ny(ts: datetime) -> datetime:
    return ts.astimezone(NY)


def session_bounds(ts: datetime) -> tuple[datetime, datetime]:
    """UTC open/close of the regular session on ts's New York date."""
    d = to_ny(ts).date()
    o = datetime.combine(d, OPEN, tzinfo=NY).astimezone(timezone.utc)
    c = datetime.combine(d, CLOSE, tzinfo=NY).astimezone(timezone.utc)
    return o, c


def in_regular_session(ts: datetime) -> bool:
    t = to_ny(ts)
    return t.weekday() < 5 and OPEN <= t.time() < CLOSE


def minutes_since_open(ts: datetime) -> float:
    o, _ = session_bounds(ts)
    return (ts - o).total_seconds() / 60.0


def minutes_to_close(ts: datetime) -> float:
    _, c = session_bounds(ts)
    return (c - ts).total_seconds() / 60.0


def filter_regular_session(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Keep bars that start at/after 9:30 and end at/before 16:00 New York time."""
    idx = df["ts"].dt.tz_convert(NY)
    start_ok = (idx.dt.hour * 60 + idx.dt.minute) >= 9 * 60 + 30
    end_min = idx.dt.hour * 60 + idx.dt.minute + minutes
    end_ok = end_min <= 16 * 60
    weekday = idx.dt.weekday < 5
    return df[start_ok & end_ok & weekday]


def session_id(ts: pd.Series) -> pd.Series:
    """New York trading date for each timestamp (used to stop labels crossing the close)."""
    return ts.dt.tz_convert(NY).dt.date


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def floor_to_bucket(ts: datetime, minutes: int) -> datetime:
    """Bucket start aligned to the 9:30 open (equivalent to clock alignment when minutes | 30)."""
    o, _ = session_bounds(ts)
    k = int((ts - o).total_seconds() // (minutes * 60))
    return o + timedelta(minutes=k * minutes)
