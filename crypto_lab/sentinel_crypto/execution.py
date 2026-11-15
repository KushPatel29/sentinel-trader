"""Shared cash/units accounting and observed-price risk controls for simulations.

No exchange order methods. Risk limits are observed-price circuit breakers;
gaps and polling delays can carry losses beyond a configured threshold.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Account:
    initial: float
    cash: float
    units: float = 0.0
    peak: float = 0.0
    day: str = ""
    day_equity: float = 0.0
    halted_day: str = ""
    halted: bool = False
    fees: float = 0.0
    turnover: float = 0.0
    cost_basis: float = 0.0
    entries: int = 0
    closed: int = 0
    wins: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    @classmethod
    def create(cls, cash: float):
        return cls(initial=cash, cash=cash, peak=cash, day_equity=cash)

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def risk(self, price: float, timestamp, cfg: dict) -> str | None:
        eq = self.equity(price)
        day = str(timestamp)[:10]
        if day != self.day:
            self.day, self.day_equity = day, eq
        self.peak = max(self.peak, eq)
        if eq <= self.peak * (1 - cfg["drawdown_limit"]):
            self.halted = True
        if eq <= self.day_equity * (1 - cfg["daily_loss_limit"]):
            self.halted_day = day
        if self.halted:
            return "drawdown_halt"
        if self.halted_day == day:
            return "daily_halt"
        return None

    def rebalance(self, target: float, mid: float, cfg: dict, *, bid=None, ask=None, multiplier=1.0) -> dict | None:
        if mid <= 0 or not np.isfinite(mid):
            raise ValueError("Invalid execution price")
        target = float(np.clip(target, 0, cfg["max_sleeve_exposure"]))
        eq = self.equity(mid)
        desired = target * eq / mid
        delta = desired - self.units
        if abs(delta * mid) < max(1e-8, eq * 0.025) and target > 0:
            return None  # Rebalance only for at least 2.5% of sleeve equity.
        if abs(delta) < 1e-12:
            return None
        fee = cfg["fee_bps"] * multiplier / 1e4
        slip = cfg["slippage_bps"] * multiplier / 1e4
        buying = delta > 0
        price = (ask or mid) * (1 + slip) if buying else (bid or mid) * (1 - slip)
        if buying:
            delta = min(delta, self.cash / (price * (1 + fee)))
            paid = delta * price * (1 + fee)
            if self.units < 1e-12:
                self.entries += 1
            self.cost_basis += paid
            self.cash -= paid
        else:
            delta = max(delta, -self.units)
            fraction = -delta / self.units
            basis = self.cost_basis * fraction
            proceeds = -delta * price * (1 - fee)
            pnl = proceeds - basis
            self.cost_basis -= basis
            self.cash += proceeds
            self.gross_profit += max(0, pnl)
            self.gross_loss += max(0, -pnl)
            if self.units + delta < 1e-10:
                self.closed += 1
                # Round-trip win rates are computed from the trade ledger, not partial sales.
        paid_fee = abs(delta) * price * fee
        self.fees += paid_fee
        self.turnover += abs(delta) * price
        self.units = max(0, self.units + delta)
        if self.cash < -1e-6:
            raise RuntimeError("Negative cash")
        return {"side": "BUY" if buying else "SELL", "units": abs(delta), "price": price,
                "fee": paid_fee, "target": target, "equity": self.equity(mid)}

    def to_dict(self):
        return asdict(self)


def simulate(px: pd.DataFrame, decisions: pd.Series, cfg: dict, *, multiplier=1.0,
             start=None, end=None, initial=1.0, risk=True) -> tuple[pd.DataFrame, list[dict], Account]:
    """t close decision executes at t+1 open. Flat start and terminal liquidation included.

    The last boundary close is used only to mark/liquidate; no same-close entry.
    """
    held = decisions.reindex(px.index).shift(1).fillna(0).clip(0, cfg["max_sleeve_exposure"])
    mask = np.ones(len(px), dtype=bool)
    if start is not None:
        mask &= px.index >= start
    if end is not None:
        mask &= px.index < end
    subset = px.loc[mask]
    if subset.empty:
        raise ValueError("Empty simulation period")
    targets = held.loc[subset.index].to_numpy()
    account = Account.create(initial)
    rows, fills = [], []
    o, c = subset.open.to_numpy(), subset.close.to_numpy()
    for i, ts in enumerate(subset.index):
        why = account.risk(float(o[i]), ts, cfg) if risk else None
        target = 0.0 if why else float(targets[i])
        fill = account.rebalance(target, float(o[i]), cfg, multiplier=multiplier)
        if fill:
            fills.append({"timestamp": str(ts), "reason": why or "signal", **fill})
        why_close = account.risk(float(c[i]), ts, cfg) if risk else None
        if why_close or i == len(subset) - 1:
            fill = account.rebalance(0.0, float(c[i]), cfg, multiplier=multiplier)
            if fill:
                fills.append({"timestamp": str(ts + pd.Timedelta(hours=4)), "reason": why_close or "end_of_test", **fill})
        eq = account.equity(float(c[i]))
        rows.append((ts, eq, account.units * c[i] / eq if eq > 0 else 0.0))
    path = pd.DataFrame(rows, columns=["timestamp", "equity", "exposure"]).set_index("timestamp")
    path["return"] = path.equity.pct_change(fill_method=None)
    path.iloc[0, path.columns.get_loc("return")] = path.equity.iloc[0] / initial - 1
    return path, fills, account
