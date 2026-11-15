"""Unlevered long/short reference simulator with explicit sessions and execution costs.

Shorts are available only for CFD research. Broker financing/contract behavior
has not been validated; these portfolios are research, never live instructions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_focus(px, decisions, cfg, spec, start, end, multiplier=1., risks=True):
    p = decisions.reindex(px.index).shift(1).fillna(0.)
    keep = (px.index >= start)&(px.index < end)
    data = px.loc[keep]
    if data.empty:
        raise ValueError("Empty execution window")
    targets = p.loc[keep].to_numpy()
    opens, closes = data.open.to_numpy(), data.close.to_numpy()
    times = data.index
    duration = pd.Timedelta(spec["interval"])
    cap = cfg["max_exposure"]
    fee = spec["fee_bps"]*multiplier/10000
    slip = spec["slippage_bps"]*multiplier/10000
    spread = spec["half_spread_bps"]*multiplier/10000
    cash, units, peak, day_eq = 1., 0., 1., 1.
    day, daily_halt, permanent = None, False, False
    rows, fills = [], []
    entries, exits, wins, overnight, cumulative_cost = 0, 0, 0, 0, 0.
    trip_equity = None

    def trade(target, price, when, reason, at_open):
        nonlocal cash, units, entries, exits, wins, trip_equity, cumulative_cost
        equity = cash+units*price
        desired = float(np.clip(target, -cap if spec["source"] == "histdata" else 0, cap))*equity/price
        if units*desired < -1e-12:
            trade(0., price, when, reason, at_open)
            trade(target, price, when, reason, at_open)
            return
        delta = desired-units
        if abs(delta*price) < max(1e-10, equity*.025) and target != 0:
            return
        if abs(delta) < 1e-12:
            return
        buying = delta > 0
        # HistData bars are BID; a buy pays a full spread. Other public OHLC is
        # treated as a mid-price approximation with a half-spread per side.
        if spec["source"] == "histdata":
            ask_col = "ask_open" if at_open else "ask_close"
            explicit_ask = float(data.loc[when, ask_col]) if ask_col in data.columns and when in data.index else None
            quoted = explicit_ask if buying and explicit_ask is not None else price*(1+2*spread) if buying else price
        else:
            quoted = price*(1+spread if buying else 1-spread)
        execution = quoted*(1+slip if buying else 1-slip)
        paid_fee = abs(delta*execution)*fee
        before = units
        if abs(before) < 1e-12:
            entries += 1
            trip_equity = equity
        cash -= delta*execution+paid_fee
        units = desired
        cumulative_cost += abs(delta*(execution-price))+paid_fee
        if abs(units) < 1e-12:
            exits += 1
            wins += int(trip_equity is not None and cash > trip_equity)
            trip_equity = None
        fills.append({"timestamp": str(when), "reason": reason, "side": "BUY" if buying else "SELL",
                      "units": float(abs(delta)), "price": float(execution), "equity": float(cash+units*price)})

    for i, ts in enumerate(times):
        op, cl = float(opens[i]), float(closes[i])
        eq = cash+units*op
        date = ts.date()
        if date != day:
            if day is not None and abs(units) > 1e-12 and spec["intraday_only"]:
                overnight += 1
            day, day_eq, daily_halt = date, eq, False
        peak = max(peak, eq)
        if risks:
            daily_halt |= eq <= day_eq*(1-cfg["daily_loss_limit"])
            permanent |= eq <= peak*(1-cfg["drawdown_limit"])
        target = float(targets[i])
        if daily_halt or permanent:
            target = 0.
        if spec["intraday_only"]:
            if not spec["decision_start_hour"]+1 <= ts.hour <= spec["last_entry_hour"]+1:
                # Existing positions may finish their holding window until 20:00.
                target = units*op/eq if units and ts.hour < spec["flatten_hour"] else 0.
            if i > 0 and ts-times[i-1] > duration*2:
                target = 0.  # A stale pre-gap signal cannot create a catch-up entry.
            if "minute_count" in data and data.minute_count.iloc[i] < 45:
                target = 0.
        trade(target, op, ts, "risk" if daily_halt or permanent else "signal", True)
        eq = cash+units*cl
        peak = max(peak, eq)
        if risks:
            daily_halt |= eq <= day_eq*(1-cfg["daily_loss_limit"])
            permanent |= eq <= peak*(1-cfg["drawdown_limit"])
        closing_session = spec["intraday_only"] and (ts+duration).hour >= spec["flatten_hour"]
        if daily_halt or permanent or closing_session or i == len(times)-1:
            trade(0., cl, ts, "session_flat" if closing_session else "risk" if daily_halt or permanent else "test_end", False)
        eq = cash+units*cl
        if eq <= 0:
            raise ValueError("Reference account equity exhausted")
        rows.append((ts, eq, abs(units*cl)/eq))
    path = pd.DataFrame(rows, columns=["timestamp", "equity", "exposure"]).set_index("timestamp")
    path["return"] = path.equity.pct_change(fill_method=None)
    path.iloc[0, path.columns.get_loc("return")] = path.equity.iloc[0]-1
    stats = {"round_trips": exits, "entries": entries, "win_rate": wins/exits if exits else None,
             "overnight_cfd_carries": overnight, "estimated_cost_fraction": cumulative_cost}
    return path, fills, stats


def focus_performance(path, spec, stats=None):
    r = path["return"].fillna(0.)
    eq = np.r_[1., path.equity.to_numpy()]
    elapsed = (path.index[-1]+pd.Timedelta(spec["interval"])-path.index[0]).total_seconds()/86400
    daily = (1+r).resample("1D").prod()-1
    sd = float(daily.std())
    m = {"total_return": float(eq[-1]-1), "cagr": float(eq[-1]**(365/max(elapsed, 1))-1),
         "sharpe": float(daily.mean()/sd*np.sqrt(365)) if sd > 1e-12 else 0.,
         "max_drawdown": float((eq/np.maximum.accumulate(eq)-1).min()),
         "mean_exposure": float(path.exposure.mean()), "calendar_days": elapsed, "bars": len(path)}
    return {**m, **(stats or {})}
