"""Executes the daily gold plan on Alpaca with GLD (paper by default).

Run once a day after the US close (GLD's session ends 16:00 New York). Each run:
  1. downloads GLD and gold-volatility (GVZ) daily bars and computes the plan on COMPLETED sessions only;
  2. reads the Alpaca position and open orders for GLD;
  3. brings the account in line with the plan:
       ENTER LONG  -> market buy for the next open (queued), sized so the initial stop costs `risk` of equity,
                      with an attached stop-loss (OTO order)
       HOLD LONG   -> keep a GTC stop order at the trailing stop; raise it, never lower it
       EXIT / STAND ASIDE with a position -> cancel the stop, sell at the next open
Nothing is sent with --dry-run. Real money needs the same three switches as the main bot."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from sentinel.gold.data import fetch_daily, split_provisional
from sentinel.gold.plan import todays_plan

log = logging.getLogger(__name__)
SYMBOL = "GLD"
GLD_CLOSE_HOUR_NY = 16


@dataclass
class Intent:
    kind: str            # "buy_oto" | "place_stop" | "raise_stop" | "sell" | "none"
    qty: float = 0
    stop: float | None = None
    note: str = ""


def load_gld(fetch=fetch_daily, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    gld = fetch(SYMBOL)
    gld = gld[gld["close"] > 0]
    gld, live = split_provisional(gld, now)
    if live["provisional"]:
        log.info("dropping the unfinished session bar (%s)", live["bar_date"].date())
    # GLD's session ends at 16:00 New York (not 17:00 like futures): re-check with the earlier cut-off
    now_ny = live["as_of"]
    if not live["provisional"] and gld.index[-1].date() == now_ny.date() and now_ny.hour < GLD_CLOSE_HOUR_NY:
        gld = gld.iloc[:-1]
    ctx = pd.DataFrame(index=gld.index)
    try:
        gvz = fetch("^GVZ")["close"]
        ctx["gvz"] = gvz.reindex(gld.index.union(gvz.index)).ffill().reindex(gld.index)
    except Exception as e:
        log.warning("GVZ unavailable (%s); using realised volatility only", e)
    return gld, ctx


def decide(plan, position_qty: float, open_stop: float | None, equity: float, buying_power: float,
           risk: float = 0.01) -> Intent:
    if plan.action == "ENTER LONG" and position_qty <= 0:
        per_share = plan.entry - plan.stop
        if per_share <= 0:
            return Intent("none", note="invalid stop")
        qty = math.floor(min(equity * risk / per_share, buying_power * 0.98 / plan.entry))
        if qty < 1:
            return Intent("none", note="position rounds to 0 shares at this risk level")
        return Intent("buy_oto", qty, round(plan.stop, 2), f"risking {risk:.1%} of equity")
    if plan.action in ("ENTER LONG", "HOLD LONG") and position_qty > 0:
        target = round(plan.stop, 2) if plan.stop else None
        if target is None:
            return Intent("none")
        if open_stop is None:
            return Intent("place_stop", position_qty, target, "protect open position")
        if target > open_stop + 0.005:
            return Intent("raise_stop", position_qty, target, f"trail stop up from {open_stop:.2f}")
        return Intent("none", note=f"stop stays at {open_stop:.2f}")
    if plan.action in ("EXIT", "STAND ASIDE") and position_qty > 0:
        return Intent("sell", position_qty, note=plan.headline)
    return Intent("none", note=plan.headline)


def run(client, risk: float = 0.01, dry_run: bool = False, fetch=fetch_daily, now=None) -> tuple[object, Intent]:
    from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import (
        GetOrdersRequest,
        MarketOrderRequest,
        ReplaceOrderRequest,
        StopLossRequest,
        StopOrderRequest,
    )

    gld, ctx = load_gld(fetch, now)
    plan = todays_plan(gld, ctx, financing=0.0)          # an ETF has no overnight financing
    acct = client.get_account()
    equity, bp = float(acct.equity), float(acct.buying_power)
    qty = 0.0
    for p in client.get_all_positions():
        if p.symbol == SYMBOL:
            qty = float(p.qty)
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[SYMBOL]))
    stops = [o for o in orders if str(getattr(o.order_type, "value", o.order_type)) == "stop"
             and str(getattr(o.side, "value", o.side)) == "sell"]
    open_stop = max((float(o.stop_price) for o in stops), default=None)
    intent = decide(plan, qty, open_stop, equity, bp, risk)
    log.info("plan %s (%s) | position %s | open stop %s -> %s %s", plan.action, plan.headline, qty, open_stop,
             intent.kind, intent.note)
    if dry_run or intent.kind == "none":
        return plan, intent
    if intent.kind == "buy_oto":
        client.submit_order(MarketOrderRequest(symbol=SYMBOL, qty=int(intent.qty), side=OrderSide.BUY,
                                               time_in_force=TimeInForce.DAY, order_class=OrderClass.OTO,
                                               stop_loss=StopLossRequest(stop_price=intent.stop)))
    elif intent.kind == "place_stop":
        client.submit_order(StopOrderRequest(symbol=SYMBOL, qty=int(intent.qty), side=OrderSide.SELL,
                                             time_in_force=TimeInForce.GTC, stop_price=intent.stop))
    elif intent.kind == "raise_stop":
        best = max(stops, key=lambda o: float(o.stop_price))
        client.replace_order_by_id(best.id, ReplaceOrderRequest(stop_price=intent.stop))
    elif intent.kind == "sell":
        for o in orders:
            client.cancel_order_by_id(o.id)
        client.submit_order(MarketOrderRequest(symbol=SYMBOL, qty=int(intent.qty), side=OrderSide.SELL,
                                               time_in_force=TimeInForce.DAY))
    return plan, intent
