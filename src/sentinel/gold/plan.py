"""Daily gold trade plan: concrete entry, stop-loss and exit levels from one fixed, tested rule.

Rule (chosen on 2006-2018 among 12 plan designs, see research/gold/plans.py):
  * Enter LONG at the next open when gold closes above both its 50-day and 200-day averages (uptrend).
  * Trailing stop-loss 2 daily volatilities below the best close since entry; it only ever moves up.
  * Exit at the stop, or at the next open after the uptrend condition breaks.
No direction forecast is involved: the rule follows the trend and cuts losses. On 2019-2026 it made money
(+7.6%/yr, less than buy & hold) with ~47% winning trades; on 2008-2018 it slightly lost. Losing trades
and losing streaks are normal and are shown with the plan."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STOP_SD = 2.0
COST = 2.5 / 1e4          # spread per round trip
SLIP = 1.0 / 1e4          # extra slippage on stop exits
FINANCING = 0.06 / 252    # CFD overnight financing per day held (long)


@dataclass
class Plan:
    action: str                 # ENTER LONG / HOLD LONG / EXIT / STAND ASIDE
    headline: str
    entry: float | None
    stop: float | None
    trigger_above: float | None   # close above this re-enters (when standing aside)
    sma50: float
    sma200: float
    trades: pd.DataFrame          # full trade history of the rule
    stats: pd.DataFrame           # per-period performance


def _sigma(gold: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    from sentinel.gold.model import daily_sigma
    return daily_sigma(gold, ctx)


def simulate(gold: pd.DataFrame, ctx: pd.DataFrame, financing: float = FINANCING) -> tuple[pd.DataFrame, dict]:
    o, lo, c = gold["open"], gold["low"], gold["close"]
    s = _sigma(gold, ctx)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    up = (c > sma50) & (c > sma200)
    trades, state = [], {"in": False}
    pending_entry = False
    for i in range(1, len(c)):
        d = c.index[i]
        if pending_entry:                                    # yesterday's close said uptrend: buy at today's open
            state = {"in": True, "entry_date": d, "entry": o.iloc[i], "best": c.iloc[i - 1], "days": 0,
                     "initial_stop": c.iloc[i - 1] * np.exp(-STOP_SD * s.iloc[i - 1])}
            pending_entry = False
        if state["in"]:
            stop = state["best"] * np.exp(-STOP_SD * s.iloc[i - 1])
            if lo.iloc[i] <= stop:                          # stopped out (a gap down fills at the open)
                px = min(stop, o.iloc[i])
                trades.append({**_close(state, d, px, "trailing stop", financing), "stop_at_exit": stop})
                state = {"in": False}
                continue
            state["days"] += 1
            state["best"] = max(state["best"], c.iloc[i])
            if not up.iloc[i] and i + 1 < len(c):            # trend broke at the close: exit at next open
                px = o.iloc[i + 1]
                trades.append({**_close(state, c.index[i + 1], px, "trend ended", financing), "stop_at_exit": stop})
                state = {"in": False}
                continue
        elif up.iloc[i] and not np.isnan(s.iloc[i]) and not np.isnan(sma200.iloc[i]):
            pending_entry = True
    T = pd.DataFrame(trades)
    state["stop_now"] = (state["best"] * np.exp(-STOP_SD * s.iloc[-1])) if state.get("in") else None
    state["pending_entry"] = pending_entry
    return T, state


def _close(state: dict, d, px: float, why: str, financing: float) -> dict:
    gross = np.log(px / state["entry"])
    net = gross - COST - (SLIP if why == "trailing stop" else 0) - financing * state["days"]
    risk = max(1e-4, 1 - state["initial_stop"] / state["entry"])       # loss fraction if the first stop is hit
    return {"entry_date": state["entry_date"], "entry": state["entry"], "exit_date": d, "exit": px, "reason": why,
            "days": state["days"], "return": np.exp(net) - 1, "initial_stop": state["initial_stop"], "risk": risk}


RISK_PER_TRADE = 0.01
MAX_LEVERAGE = 3.0


def period_stats(T: pd.DataFrame, a: str, b: str) -> dict:
    t = T[(T["entry_date"] >= a) & (T["entry_date"] <= b)]
    if t.empty:
        return {}
    r = t["return"]
    # account-level result when every trade is sized so the initial stop costs 1% of the account
    f = (RISK_PER_TRADE / t["risk"]).clip(upper=MAX_LEVERAGE)
    acct = (1 + f * r).cumprod()
    eq = (1 + r).cumprod()
    years = (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25
    streak = worst = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        worst = max(worst, streak)
    return {"period": f"{a[:4]}–{b[:4]}", "trades": len(t), "win_rate": float((r > 0).mean()),
            "avg_win": float(r[r > 0].mean()), "avg_loss": float(r[r < 0].mean()),
            "annual_return": float(eq.iloc[-1] ** (1 / years) - 1), "worst_trade": float(r.min()),
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "worst_losing_streak": worst,
            "avg_days_held": float(t["days"].mean()),
            "account_return_1pct_risk": float(acct.iloc[-1] ** (1 / years) - 1),
            "account_drawdown_1pct_risk": float((acct / acct.cummax() - 1).min())}


def todays_plan(gold: pd.DataFrame, ctx: pd.DataFrame, financing: float = FINANCING) -> Plan:
    T, st = simulate(gold, ctx, financing)
    c = gold["close"]
    sma50, sma200 = float(c.rolling(50).mean().iloc[-1]), float(c.rolling(200).mean().iloc[-1])
    s = float(_sigma(gold, ctx).iloc[-1])
    last = float(c.iloc[-1])
    end = str(gold.index[-1].date())
    stats = pd.DataFrame([x for x in (period_stats(T, "2008-01-01", "2018-12-31"),
                                      period_stats(T, "2019-01-01", end)) if x])
    if st.get("pending_entry"):
        stop = last * np.exp(-STOP_SD * s)
        return Plan("ENTER LONG", "Uptrend confirmed at the close: buy at the next open", last, stop, None, sma50,
                    sma200, T, stats)
    if st.get("in"):
        return Plan("HOLD LONG", "In a long trade: keep the trailing stop", float(st["entry"]), float(st["stop_now"]),
                    None, sma50, sma200, T, stats)
    if T is not None and len(T) and pd.Timestamp(T["exit_date"].iloc[-1]) >= gold.index[-1]:
        return Plan("EXIT", f"Trade closed today ({T['reason'].iloc[-1]})", None, None, max(sma50, sma200), sma50,
                    sma200, T, stats)
    return Plan("STAND ASIDE", "No uptrend: no long position", None, None, max(sma50, sma200), sma50, sma200, T, stats)
