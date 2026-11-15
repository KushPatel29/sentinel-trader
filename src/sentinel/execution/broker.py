"""Broker adapters. The engine only talks to the `Broker` interface, so the same trading logic runs
against the simulator (backtest/replay) and Alpaca (paper/live)."""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

from sentinel.config import Config
from sentinel.events import AccountState, Bar, ClockState, OrderAck, PositionState
from sentinel.market_time import session_bounds

log = logging.getLogger(__name__)


class Broker(ABC):
    latency_cb = None  # callable(ms, what) set by the engine

    @abstractmethod
    async def get_account(self) -> AccountState: ...

    @abstractmethod
    async def get_positions(self) -> dict[str, PositionState]: ...

    @abstractmethod
    async def submit_bracket(self, symbol: str, qty: int, take_profit: float, stop_loss: float,
                             client_order_id: str) -> OrderAck: ...

    @abstractmethod
    async def close_position(self, symbol: str, reason: str) -> None: ...

    @abstractmethod
    async def close_all(self, reason: str) -> None: ...

    @abstractmethod
    async def get_clock(self) -> ClockState: ...


# ----------------------------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------------------------
@dataclass
class SimPosition:
    qty: int
    avg_price: float
    take_profit: float
    stop_loss: float
    entry_ts: datetime
    client_order_id: str


@dataclass
class SimFill:
    ts: datetime
    symbol: str
    side: str
    qty: int
    price: float
    reason: str
    pnl: float | None = None


class SimBroker(Broker):
    """Fills market entries at the latest close +/- slippage (the engine decides at a bar's close),
    then works the bracket against later bars' highs/lows. Gaps through a stop fill at the open.
    If a bar touches both legs, the stop is assumed to fill first (conservative)."""

    def __init__(self, cfg: Config, initial_cash: float):
        self.cfg = cfg
        self.cash = initial_cash
        self.positions: dict[str, SimPosition] = {}
        self.last: dict[str, float] = {}
        self.fills: list[SimFill] = []
        self.now: datetime | None = None
        self.slip = cfg.execution.slippage_bps / 1e4
        self.comm = cfg.execution.commission_per_share
        self.fees_paid = 0.0
        self.day_trades: list[datetime] = []
        self._seen_ids: set[str] = set()

    # --- market data hooks driven by the backtest / replay loop ---
    def process_bar(self, bar: Bar) -> list[SimFill]:
        out = []
        pos = self.positions.get(bar.symbol)
        if pos is not None and bar.ts >= pos.entry_ts:
            px, why = None, ""
            if bar.open <= pos.stop_loss:
                px, why = bar.open, "stop (gap)"
            elif bar.low <= pos.stop_loss:
                px, why = pos.stop_loss, "stop"
            elif bar.open >= pos.take_profit:
                px, why = bar.open, "take profit (gap)"
            elif bar.high >= pos.take_profit:
                px, why = pos.take_profit, "take profit"
            if px is not None:
                if why.startswith("stop"):
                    px *= 1 - self.slip
                out.append(self._sell(bar.symbol, px, why, bar.end))
        self.last[bar.symbol] = bar.close
        return out

    def _sell(self, symbol: str, price: float, reason: str, ts: datetime) -> SimFill:
        pos = self.positions.pop(symbol)
        fee = self.comm * pos.qty
        self.cash += pos.qty * price - fee
        self.fees_paid += fee
        pnl = (price - pos.avg_price) * pos.qty - fee
        if pos.entry_ts.date() == ts.date():
            self.day_trades.append(ts)
        f = SimFill(ts, symbol, "sell", pos.qty, price, reason, pnl)
        self.fills.append(f)
        return f

    # --- Broker interface ---
    async def get_account(self) -> AccountState:
        eq = self.equity()
        recent = [t for t in self.day_trades if self.now and t > self.now - timedelta(days=7)]
        return AccountState(equity=eq, cash=self.cash, buying_power=max(0.0, self.cash), daytrade_count=len(recent))

    def equity(self) -> float:
        return self.cash + sum(p.qty * self.last.get(s, p.avg_price) for s, p in self.positions.items())

    async def get_positions(self) -> dict[str, PositionState]:
        return {s: PositionState(s, p.qty, p.avg_price, p.qty * self.last.get(s, p.avg_price))
                for s, p in self.positions.items()}

    async def submit_bracket(self, symbol: str, qty: int, take_profit: float, stop_loss: float,
                             client_order_id: str) -> OrderAck:
        if client_order_id in self._seen_ids:
            return OrderAck(client_order_id, symbol, qty, "buy", "rejected_duplicate")
        self._seen_ids.add(client_order_id)
        if symbol in self.positions or qty <= 0:
            return OrderAck(client_order_id, symbol, qty, "buy", "rejected")
        px = self.last[symbol] * (1 + self.slip)
        fee = self.comm * qty
        self.cash -= qty * px + fee
        self.fees_paid += fee
        self.positions[symbol] = SimPosition(qty, px, take_profit, stop_loss, self.now, client_order_id)
        self.fills.append(SimFill(self.now, symbol, "buy", qty, px, "entry"))
        return OrderAck(client_order_id, symbol, qty, "buy", "filled", filled_price=px,
                        take_profit=take_profit, stop_loss=stop_loss)

    async def close_position(self, symbol: str, reason: str) -> None:
        if symbol in self.positions:
            self._sell(symbol, self.last[symbol] * (1 - self.slip), reason, self.now)

    async def close_all(self, reason: str) -> None:
        for s in list(self.positions):
            await self.close_position(s, reason)

    async def get_clock(self) -> ClockState:
        o, c = session_bounds(self.now)
        return ClockState(now=self.now, is_open=o <= self.now < c, next_open=o, next_close=c)


# ----------------------------------------------------------------------------------------------
# Alpaca
# ----------------------------------------------------------------------------------------------
class AlpacaBroker(Broker):
    """Thin async wrapper over alpaca-py's synchronous TradingClient. Every call runs in a worker
    thread and is timed; timings go to the risk officer's latency monitor."""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        from alpaca.trading.client import TradingClient

        self.paper = paper
        self.client = TradingClient(api_key, secret_key, paper=paper)

    async def _call(self, what: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        finally:
            ms = (time.perf_counter() - t0) * 1000
            if self.latency_cb:
                self.latency_cb(ms, what)

    async def get_account(self) -> AccountState:
        a = await self._call("get_account", self.client.get_account)
        return AccountState(equity=float(a.equity), cash=float(a.cash), buying_power=float(a.buying_power),
                            daytrade_count=int(a.daytrade_count or 0),
                            pattern_day_trader=bool(a.pattern_day_trader),
                            trading_blocked=bool(a.trading_blocked or a.account_blocked))

    async def get_positions(self) -> dict[str, PositionState]:
        ps = await self._call("get_positions", self.client.get_all_positions)
        return {p.symbol: PositionState(p.symbol, float(p.qty), float(p.avg_entry_price), float(p.market_value or 0))
                for p in ps}

    async def submit_bracket(self, symbol: str, qty: int, take_profit: float, stop_loss: float,
                             client_order_id: str) -> OrderAck:
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET, client_order_id=client_order_id,
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
        )
        try:
            o = await self._call("submit_order", self.client.submit_order, order_data=req)
        except Exception as e:  # APIError: duplicate client_order_id, insufficient buying power, halted...
            log.error("order rejected for %s: %s", symbol, e)
            return OrderAck(client_order_id, symbol, qty, "buy", "rejected", extra={"error": str(e)})
        return OrderAck(client_order_id, symbol, qty, "buy", str(o.status.value if hasattr(o.status, "value")
                        else o.status), broker_id=str(o.id), take_profit=round(take_profit, 2),
                        stop_loss=round(stop_loss, 2))

    async def _cancel_symbol_orders(self, symbol: str) -> None:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = await self._call("get_orders", self.client.get_orders,
                                  filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol], nested=True))
        for o in orders:
            try:
                await self._call("cancel_order", self.client.cancel_order_by_id, o.id)
            except Exception as e:
                log.warning("cancel %s failed: %s", o.id, e)

    async def close_position(self, symbol: str, reason: str) -> None:
        # Bracket legs reserve the shares, so they must be cancelled before the position can be closed.
        await self._cancel_symbol_orders(symbol)
        try:
            await self._call("close_position", self.client.close_position, symbol)
            log.info("closed %s (%s)", symbol, reason)
        except Exception as e:
            log.error("close %s failed: %s", symbol, e)

    async def close_all(self, reason: str) -> None:
        await self._call("close_all", self.client.close_all_positions, cancel_orders=True)
        log.warning("closed all positions (%s)", reason)

    async def get_clock(self) -> ClockState:
        c = await self._call("get_clock", self.client.get_clock)
        return ClockState(now=c.timestamp, is_open=bool(c.is_open), next_open=c.next_open, next_close=c.next_close)
