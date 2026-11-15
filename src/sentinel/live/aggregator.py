"""Builds N-minute bars from the 1-minute bar stream, with boundaries aligned to the 9:30 open
(identical to Alpaca's historical N-minute bars when N divides 30)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sentinel.events import Bar
from sentinel.market_time import floor_to_bucket, in_regular_session


@dataclass
class _Partial:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_bar(self, symbol: str, minutes: int) -> Bar:
        return Bar(symbol, self.start, self.open, self.high, self.low, self.close, self.volume, minutes)


class BarAggregator:
    def __init__(self, minutes: int):
        self.minutes = minutes
        self.partials: dict[str, _Partial] = {}
        self.last_emitted: dict[str, datetime] = {}

    def add(self, b: Bar) -> list[Bar]:
        """Add a 1-minute bar; returns any N-minute bars it completes."""
        if not in_regular_session(b.ts):
            return []
        out: list[Bar] = []
        start = floor_to_bucket(b.ts, self.minutes)
        if self.last_emitted.get(b.symbol) and start <= self.last_emitted[b.symbol]:
            return []                                    # late minute for an already-emitted bucket
        p = self.partials.get(b.symbol)
        if p is not None and start > p.start:            # a minute from the next bucket: close the old one
            out.append(self._emit(b.symbol))
            p = None
        if p is None:
            self.partials[b.symbol] = _Partial(start, b.open, b.high, b.low, b.close, b.volume)
        else:
            p.high, p.low = max(p.high, b.high), min(p.low, b.low)
            p.close = b.close
            p.volume += b.volume
        if b.ts + timedelta(minutes=1) >= start + timedelta(minutes=self.minutes):
            out.append(self._emit(b.symbol))             # last minute of the bucket arrived
        return out

    def flush_due(self, now: datetime, grace_s: float = 20.0) -> list[Bar]:
        """Emit buckets whose end has passed (covers minutes with no trades on the IEX feed)."""
        out = []
        for sym in list(self.partials):
            p = self.partials[sym]
            if now >= p.start + timedelta(minutes=self.minutes, seconds=grace_s):
                out.append(self._emit(sym))
        return out

    def _emit(self, sym: str) -> Bar:
        p = self.partials.pop(sym)
        self.last_emitted[sym] = p.start
        return p.to_bar(sym, self.minutes)
