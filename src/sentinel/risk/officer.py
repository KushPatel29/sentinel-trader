"""The risk officer: an independent layer with veto power over every entry.

It never generates trades. It can only block entries, or order the engine to flatten and halt.
Rules:
  * daily drawdown cap  -> flatten everything, no new entries until the next session
  * kill-switch file    -> flatten everything, stop the bot (sticky until restart)
  * broker latency      -> block entries after N consecutive slow API calls
  * stale market data   -> ignore signals from bars that arrive too late
  * exposure limits     -> max open positions, max gross exposure
  * session timing      -> no entries right after the open / near the close; flatten before close
  * account flags       -> trading blocked; optional legacy pattern-day-trader limit"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from sentinel.config import Config
from sentinel.events import AccountState

log = logging.getLogger(__name__)


@dataclass
class RiskEvent:
    ts: datetime
    level: str
    rule: str
    message: str


@dataclass
class RiskState:
    session: date | None = None
    day_start_equity: float = 0.0
    peak_equity_today: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    killed: bool = False
    latency_strikes: int = 0
    events: list[RiskEvent] = field(default_factory=list)


class RiskOfficer:
    def __init__(self, cfg: Config, on_event=None):
        self.cfg = cfg
        self.rc = cfg.risk
        self.state = RiskState()
        self._on_event = on_event

    def _emit(self, ts: datetime, level: str, rule: str, msg: str) -> None:
        ev = RiskEvent(ts, level, rule, msg)
        self.state.events.append(ev)
        (log.warning if level != "info" else log.info)("[risk:%s] %s", rule, msg)
        if self._on_event:
            self._on_event(ev)

    # ---------- session lifecycle ----------
    def new_session(self, session: date, equity: float, ts: datetime) -> None:
        if self.state.session == session:
            return
        was_halted = self.state.halted
        self.state.session = session
        self.state.day_start_equity = equity
        self.state.peak_equity_today = equity
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.latency_strikes = 0
        self._emit(ts, "info", "session", f"new session {session}, start equity {equity:,.2f}"
                   + (" (daily halt cleared)" if was_halted else ""))

    # ---------- continuous monitoring ----------
    def on_equity(self, equity: float, ts: datetime) -> bool:
        """Returns True if the engine must flatten now."""
        s = self.state
        if s.day_start_equity <= 0:
            return False
        s.peak_equity_today = max(s.peak_equity_today, equity)
        dd = 1 - equity / s.day_start_equity
        if not s.halted and dd >= self.rc.daily_drawdown_halt:
            s.halted = True
            s.halt_reason = f"daily drawdown {dd:.2%} >= {self.rc.daily_drawdown_halt:.2%}"
            self._emit(ts, "critical", "daily_drawdown", s.halt_reason + "; flattening and halting for the day")
            return True
        return False

    def on_latency(self, ms: float, ts: datetime, what: str = "api") -> None:
        if ms > self.rc.max_api_latency_ms:
            self.state.latency_strikes += 1
            if self.state.latency_strikes == self.rc.latency_strikes:
                self._emit(ts, "warning", "latency", f"{what} took {ms:.0f} ms; "
                           f"{self.state.latency_strikes} slow calls in a row, blocking entries")
        elif self.state.latency_strikes:
            if self.state.latency_strikes >= self.rc.latency_strikes:
                self._emit(ts, "info", "latency", "broker latency recovered; entries re-enabled")
            self.state.latency_strikes = 0

    def bar_is_stale(self, bar_end: datetime, now: datetime) -> bool:
        delay = (now - bar_end).total_seconds()
        if delay > self.rc.max_bar_delay_s:
            self._emit(now, "warning", "stale_data", f"bar ended {delay:.0f}s ago; ignoring its signal")
            return True
        return False

    def kill_switch(self, ts: datetime) -> bool:
        if self.state.killed:
            return True
        if Path(self.rc.kill_switch_file).exists():
            self.state.killed = True
            self.state.halted = True
            self.state.halt_reason = "kill switch"
            self._emit(ts, "critical", "kill_switch", f"'{self.rc.kill_switch_file}' found; flattening and stopping")
            return True
        return False

    def should_flatten_eod(self, minutes_to_close: float) -> bool:
        return minutes_to_close <= self.rc.flatten_minutes_before_close

    # ---------- pre-trade gate ----------
    def entry_check(self, *, minutes_since_open: float, minutes_to_close: float, open_positions: int,
                    gross_exposure: float, equity: float, new_notional: float,
                    account: AccountState | None = None) -> tuple[bool, str]:
        s, rc = self.state, self.rc
        if s.killed or s.halted:
            return False, f"halted: {s.halt_reason}"
        if s.latency_strikes >= rc.latency_strikes:
            return False, "broker latency"
        if minutes_since_open < rc.no_entry_minutes_after_open:
            return False, "too soon after open"
        if minutes_to_close < rc.no_entry_minutes_before_close:
            return False, "too close to the close"
        if open_positions >= rc.max_open_positions:
            return False, f"max open positions ({rc.max_open_positions})"
        if equity <= 0 or (gross_exposure + new_notional) / equity > rc.max_gross_exposure:
            return False, "gross exposure limit"
        if account is not None:
            if account.trading_blocked:
                return False, "account trading blocked"
            if rc.pdt_guard and account.equity < 25_000 and account.daytrade_count >= 3:
                return False, "pattern-day-trader guard (sub-$25k account at 3 day trades)"
        return True, "ok"
