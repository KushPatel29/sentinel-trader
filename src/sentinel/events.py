"""Plain data types passed between layers. Timestamps are always tz-aware UTC."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


@dataclass(frozen=True)
class Bar:
    symbol: str
    ts: datetime          # bar START time (Alpaca convention)
    open: float
    high: float
    low: float
    close: float
    volume: float
    minutes: int = 1      # bar length

    @property
    def end(self) -> datetime:
        """Moment the bar is complete and its information is usable."""
        return self.ts + timedelta(minutes=self.minutes)


@dataclass(frozen=True)
class NewsItem:
    id: str
    ts: datetime
    headline: str
    summary: str = ""
    symbols: tuple[str, ...] = ()
    source: str = ""


@dataclass(frozen=True)
class SentimentScore:
    news_id: str
    score: float          # -1 (very negative) .. +1 (very positive)
    engine: str
    rationale: str = ""


class Action(str, Enum):
    HOLD = "hold"
    ENTER_LONG = "enter_long"
    EXIT = "exit"


@dataclass
class Decision:
    action: Action
    reason: str
    size_mult: float = 1.0


@dataclass
class SignalSnapshot:
    symbol: str
    ts: datetime
    ready: bool
    prob: float = float("nan")
    atr: float = float("nan")
    close: float = float("nan")
    regime: int = -1
    regime_blocked: bool = False


@dataclass
class AccountState:
    equity: float
    cash: float
    buying_power: float
    daytrade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False


@dataclass
class PositionState:
    symbol: str
    qty: float
    avg_price: float
    market_value: float


@dataclass
class OrderAck:
    client_order_id: str
    symbol: str
    qty: float
    side: str
    status: str
    broker_id: str = ""
    filled_price: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ClockState:
    now: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime
