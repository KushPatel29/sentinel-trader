"""TradingEngine: wires signals, sentiment, risk and execution together.

One engine class drives backtests, replays and live trading. Only three things change between
modes: where bars/news come from, which signal provider is used, and which broker fills orders."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from sentinel.config import Config
from sentinel.data.store import Store
from sentinel.events import AccountState, Action, Bar, ClockState, Decision, NewsItem, PositionState
from sentinel.execution.broker import Broker
from sentinel.market_time import minutes_since_open, session_bounds, to_ny
from sentinel.models.sentiment import SentimentAnalyzer, SentimentState
from sentinel.risk.officer import RiskOfficer
from sentinel.risk.sizing import position_size
from sentinel.strategy.ensemble import decide

log = logging.getLogger(__name__)


@dataclass
class EquityPoint:
    ts: datetime
    equity: float
    cash: float
    gross: float
    n_positions: int


class TradingEngine:
    def __init__(self, cfg: Config, broker: Broker, signals, sentiment: SentimentState, risk: RiskOfficer,
                 analyzer: SentimentAnalyzer | None = None, store: Store | None = None,
                 react_to_news_immediately: bool = False, clock_ttl_s: float = 0.0,
                 timer_step_minutes: float = 0.0):
        self.cfg = cfg
        self.broker = broker
        self.signals = signals
        self.sentiment = sentiment
        self.risk = risk
        self.analyzer = analyzer
        self.store = store
        self.react_now = react_to_news_immediately
        self.timer_step_minutes = timer_step_minutes
        self.managed = set(cfg.symbols)
        self.positions: dict[str, PositionState] = {}
        self.account: AccountState | None = None
        self.clock: ClockState | None = None
        self._clock_at: datetime | None = None
        self._clock_ttl = clock_ttl_s
        self.equity_curve: list[EquityPoint] = []
        self.decisions: list[tuple[datetime, str, str, str]] = []
        self.stopped = False
        self._lock = asyncio.Lock()   # serialises order flow between bar, news and timer tasks
        broker.latency_cb = lambda ms, what: self.risk.on_latency(ms, self._now or datetime.now().astimezone(), what)
        self._now: datetime | None = None

    # ------------------------------------------------------------------ state sync
    async def refresh(self) -> None:
        # Only positions in the configured universe are the bot's to manage. Anything else in the
        # account is ignored (but a dedicated paper account is still strongly recommended).
        allp = await self.broker.get_positions()
        self.positions = {s: p for s, p in allp.items() if s in self.managed}
        self.account = await self.broker.get_account()

    async def flatten(self, reason: str) -> None:
        for sym in list(self.positions):
            await self.broker.close_position(sym, reason)
        await self.refresh()

    async def _get_clock(self, now: datetime) -> ClockState:
        if (self.clock is None or self._clock_at is None
                or (now - self._clock_at).total_seconds() >= self._clock_ttl):
            self.clock = await self.broker.get_clock()
            self._clock_at = now
        return self.clock

    async def _minutes_open_close(self, now: datetime) -> tuple[float, float]:
        clock = await self._get_clock(now)
        o, c = session_bounds(now)
        if clock.is_open and clock.next_close:        # broker clock knows about half-days
            c = min(c, clock.next_close)
        return minutes_since_open(now), (c - now).total_seconds() / 60.0

    def _gross(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    # ------------------------------------------------------------------ events
    async def on_timer(self, now: datetime) -> None:
        """Periodic housekeeping: kill switch, new session, drawdown halt, end-of-day flatten."""
        self._now = now
        if self.stopped:
            return
        async with self._lock:
            await self.refresh()
            if self.risk.kill_switch(now):
                await self.flatten("kill switch")
                self.stopped = True
                return
            o, c = session_bounds(now)
            if o <= now <= c:
                self.risk.new_session(to_ny(now).date(), self.account.equity, now)
            if self.risk.on_equity(self.account.equity, now) and self.positions:
                await self.flatten("daily drawdown halt")
            _, to_close = await self._minutes_open_close(now)
            # With a discrete timer (one tick per bar in backtests/replays) flatten at the last tick
            # before the cut-off instead of overshooting it.
            to_close_next_tick = to_close - self.timer_step_minutes
            if self.positions and o <= now <= c and (self.risk.should_flatten_eod(to_close)
                                                     or self.risk.should_flatten_eod(to_close_next_tick)):
                await self.flatten("end-of-day flatten")
            pt = EquityPoint(now, self.account.equity, self.account.cash, self._gross(), len(self.positions))
            self.equity_curve.append(pt)
            if self.store:
                self.store.log_equity(now, pt.equity, pt.cash, pt.gross, pt.n_positions)

    async def on_news(self, item: NewsItem, now: datetime, score: float | None = None) -> float:
        self._now = now
        if score is None:
            if self.analyzer is None:
                return 0.0
            res = await asyncio.to_thread(self.analyzer.score, item)   # FinBERT/LLM must not block the loop
            score = res.score
            if self.store:
                self.store.upsert_news([item])
                self.store.upsert_sentiment(item.id, res.engine, res.score, res.rationale)
        self.sentiment.add(item.symbols, item.ts, score)
        if self.react_now:
            for sym in item.symbols:
                if sym in self.positions and self.sentiment.score(sym, now) <= self.cfg.sentiment.veto_threshold:
                    async with self._lock:
                        log.warning("news exit %s: %s", sym, item.headline)
                        await self.broker.close_position(sym, f"news veto: {item.headline[:80]}")
                        await self.refresh()
                        self._log_decision(now, sym, float("nan"), -1, score, "exit", "news veto (live)")
        return score

    async def on_bar(self, bar: Bar, now: datetime) -> Decision:
        self._now = now
        sig = self.signals.on_bar(bar)                    # always update windows, even when halted
        if self.stopped:
            return Decision(Action.HOLD, "stopped")
        sent = self.sentiment.score(bar.symbol, bar.end)
        d = decide(sig, sent, bar.symbol in self.positions, self.cfg)

        if d.action == Action.ENTER_LONG:
            d = await self._try_enter(bar, sig, d, now)
        elif d.action == Action.EXIT:
            async with self._lock:
                await self.broker.close_position(bar.symbol, d.reason)
                await self.refresh()
        self._log_decision(now, bar.symbol, sig.prob, sig.regime, sent, d.action.value, d.reason)
        return d

    async def _try_enter(self, bar: Bar, sig, d: Decision, now: datetime) -> Decision:
        if self.risk.bar_is_stale(bar.end, now):
            return Decision(Action.HOLD, "stale bar")
        async with self._lock:
            await self.refresh()
            acct = self.account
            since_open, to_close = await self._minutes_open_close(now)
            headroom = acct.equity * self.cfg.risk.max_gross_exposure - self._gross()
            qty = position_size(acct.equity, sig.close, sig.atr, sig.prob, self.cfg, d.size_mult,
                                available_notional=min(headroom, acct.buying_power))
            if qty < 1:
                return Decision(Action.HOLD, d.reason + " | size rounds to 0")
            ok, why = self.risk.entry_check(
                minutes_since_open=since_open, minutes_to_close=to_close, open_positions=len(self.positions),
                gross_exposure=self._gross(), equity=acct.equity, new_notional=qty * sig.close, account=acct)
            if not ok:
                return Decision(Action.HOLD, d.reason + f" | risk: {why}")
            tp = sig.close + self.cfg.labels.tp_atr * sig.atr
            sl = sig.close - self.cfg.labels.sl_atr * sig.atr
            cid = f"snt-{bar.symbol}-{bar.ts:%Y%m%d%H%M}"   # idempotent: one entry per symbol per bar
            ack = await self.broker.submit_bracket(bar.symbol, qty, tp, sl, cid)
            if self.store:
                self.store.log_order(now, ack)
            await self.refresh()
            if ack.status.startswith("rejected"):
                return Decision(Action.HOLD, d.reason + f" | order {ack.status}")
            log.info("ENTER %s x%d @~%.2f tp=%.2f sl=%.2f (%s)", bar.symbol, qty, sig.close, tp, sl, d.reason)
            return Decision(Action.ENTER_LONG, d.reason + f" | qty {qty}", d.size_mult)

    def _log_decision(self, ts, symbol, prob, regime, sent, action, reason) -> None:
        self.decisions.append((ts, symbol, action, reason))
        if self.store:
            self.store.log_signal(ts, symbol, prob, regime, sent, action, reason)

    async def shutdown(self, flatten: bool) -> None:
        if flatten:
            await self.refresh()
            await self.flatten("shutdown")
        self.stopped = True
