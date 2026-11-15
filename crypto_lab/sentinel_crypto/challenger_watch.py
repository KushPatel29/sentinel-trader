"""New real-quote paper books for every frozen challenger. No account/order API."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import time

import joblib
import pandas as pd

from .challenger_research import load_challengers
from .config import ROOT, atomic_json, digest
from .focus_features import build_features
from .public_live import quote_snapshot, recent_bars


def database(root=ROOT):
    folder = root / "runtime" / "challengers"
    folder.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(folder / "paper.sqlite", timeout=15)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS forecasts (
          symbol TEXT, model TEXT, bar TEXT, observed TEXT, due TEXT, probability REAL,
          entry_mid REAL, action TEXT, notional REAL, entry_fill REAL, entry_fee REAL,
          exit_mid REAL, exit_fill REAL, exit_fee REAL, paper_pnl REAL,
          actual_up INTEGER, label_state TEXT DEFAULT 'pending', closed TEXT,
          lateness_seconds REAL, PRIMARY KEY(symbol,model,bar));
        CREATE TABLE IF NOT EXISTS events(time TEXT,symbol TEXT,state TEXT,message TEXT);
        CREATE TABLE IF NOT EXISTS quotes(symbol TEXT,observed TEXT,bid REAL,ask REAL,PRIMARY KEY(symbol,observed));
        CREATE TABLE IF NOT EXISTS startup_diagnostics AS SELECT * FROM forecasts WHERE 0;
        CREATE UNIQUE INDEX IF NOT EXISTS diagnostic_identity ON startup_diagnostics(symbol,model,bar);
    """)
    return con


def checked_table(table):
    if table not in ("forecasts", "startup_diagnostics"):
        raise ValueError("Unknown paper cohort")
    return table


def mark_outcomes(con, symbol, quote, now, cfg, stopping=False, table="forecasts"):
    table = checked_table(table)
    records = con.execute(
        f"SELECT model,bar,due,entry_mid,action,notional,entry_fill,entry_fee FROM {table} WHERE symbol=? AND label_state='pending'",
        (symbol,),
    ).fetchall()
    for model, bar, due, entry, action, notional, buy, fee in records:
        if not stopping and now < pd.Timestamp(due):
            continue
        lateness = (now - pd.Timestamp(due)).total_seconds()
        state = (
            "cancelled_stop"
            if stopping
            else "observed"
            if lateness <= cfg["maximum_label_delay_seconds"]
            else "expired_outage"
        )
        # Delayed paper positions close at the actual newly observed bid, never a
        # fabricated scheduled price. Their expired direction labels are excluded.
        sell = quote["bid"] * (1 - cfg["paper_slippage_bps"] / 10000) if action == "long" else None
        exit_fee = notional / buy * sell * cfg["paper_fee_bps"] / 10000 if action == "long" else 0.0
        pnl = notional / buy * sell - notional - fee - exit_fee if action == "long" else 0.0
        actual = int(quote["mid"] > entry) if state == "observed" else None
        con.execute(
            f"UPDATE {table} SET exit_mid=?,exit_fill=?,exit_fee=?,paper_pnl=?,actual_up=?,label_state=?,closed=?,lateness_seconds=? WHERE symbol=? AND model=? AND bar=?",
            (quote["mid"], sell, exit_fee, pnl, actual, state, str(now), lateness, symbol, model, bar),
        )


def summary(con, cfg, table="forecasts"):
    frame = pd.read_sql_query(f"SELECT * FROM {checked_table(table)}", con)
    output = []
    for (symbol, model), part in frame.groupby(["symbol", "model"]):
        observed = part[part.label_state == "observed"]
        pnl = float(part.paper_pnl.fillna(0).sum())
        output.append(
            {
                "symbol": symbol,
                "model": model,
                "forecasts": len(part),
                "observed_outcomes": len(observed),
                "pending": int((part.label_state == "pending").sum()),
                "expired": int((part.label_state == "expired_outage").sum()),
                "accuracy": float(((observed.probability >= 0.5) == observed.actual_up).mean())
                if len(observed)
                else None,
                "brier": float(((observed.probability - observed.actual_up) ** 2).mean()) if len(observed) else None,
                "closed_paper_trades": int(((part.action == "long") & part.closed.notna()).sum()),
                "realized_paper_pnl": pnl,
                "paper_balance": cfg["paper_initial_balance"] + pnl,
            }
        )
    return output


def probabilities(policy, px, cfg):
    f, _ = build_features(px, policy["spec"], {})
    values = {name: float(m.predict(f)[0]) for name, m in policy["members"].items()}
    values["prior"] = policy["members"][cfg["models"][0]].prior
    values["hybrid"] = cfg["hybrid_prior_weight"] * values["prior"] + (1 - cfg["hybrid_prior_weight"]) * sum(
        values[n] * w for n, w in policy["weights"].items()
    )
    return values


def record(con, symbol, quote, now, bar, values, cfg, table="forecasts"):
    table = checked_table(table)
    for name, p in values.items():
        if not 0 <= p <= 1:
            raise ValueError("Invalid model probability")
        # Independent virtual books and cohorts; never sum as a portfolio.
        pnl = con.execute(
            f"SELECT COALESCE(SUM(paper_pnl),0) FROM {table} WHERE symbol=? AND model=?", (symbol, name)
        ).fetchone()[0]
        balance = cfg["paper_initial_balance"] + pnl
        action = "long" if p >= cfg["entry_threshold"] and balance > 0 else "cash"
        notional = balance * cfg["paper_exposure"] if action == "long" else 0.0
        fill = quote["ask"] * (1 + cfg["paper_slippage_bps"] / 10000) if action == "long" else None
        fee = notional * cfg["paper_fee_bps"] / 10000
        con.execute(
            f"INSERT OR IGNORE INTO {table}(symbol,model,bar,observed,due,probability,entry_mid,action,notional,entry_fill,entry_fee,label_state) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                symbol,
                name,
                str(bar),
                str(now),
                str(now + pd.Timedelta(hours=24)),
                p,
                quote["mid"],
                action,
                notional,
                fill,
                fee,
                "pending",
            ),
        )


def observe(con, policies, histories, cfg, root=ROOT, quotes=None, include_startup=True):
    injected = quotes is not None
    quotes = quotes if quotes is not None else quote_snapshot()
    now = pd.Timestamp(quotes[0]["observed_utc"])
    with con:
        con.executemany(
            "INSERT OR IGNORE INTO quotes VALUES(?,?,?,?)",
            [(q["symbol"], str(now), q["bid"], q["ask"]) for q in quotes],
        )
    states = {}
    for quote in quotes:
        symbol = quote["symbol"]
        if symbol not in policies:
            continue
        try:
            with con:
                mark_outcomes(con, symbol, quote, now, cfg)
                mark_outcomes(con, symbol, quote, now, cfg, table="startup_diagnostics")
                policy = policies[symbol]
                if now - pd.Timestamp(policy["boundary"]) > pd.Timedelta(days=cfg["maximum_model_age_days"]):
                    states[symbol] = {"state": "MODEL_STALE_NO_NEW_FORECAST"}
                    continue
                expected = now.floor("4h")
                if quote["spread_bps"] > 15:
                    raise ValueError("Spread quality limit exceeded")
                if (
                    include_startup
                    and not con.execute(
                        "SELECT 1 FROM startup_diagnostics WHERE symbol=? LIMIT 1", (symbol,)
                    ).fetchone()
                ):
                    if symbol not in histories:
                        histories[symbol], _ = recent_bars(symbol)
                    input_close = histories[symbol].index[-1] + pd.Timedelta(hours=4)
                    if not pd.Timedelta(0) <= now - input_close <= pd.Timedelta(hours=4):
                        raise ValueError("Startup inputs are stale or in the future")
                    values = probabilities(policy, histories[symbol], cfg)
                    entry = quote if injected else next(q for q in quote_snapshot() if q["symbol"] == symbol)
                    if entry["spread_bps"] > 15:
                        raise ValueError("Fresh entry quote spread limit exceeded")
                    record(
                        con,
                        symbol,
                        entry,
                        pd.Timestamp(entry["observed_utc"]),
                        histories[symbol].index[-1],
                        values,
                        cfg,
                        table="startup_diagnostics",
                    )
                if now - expected > pd.Timedelta(seconds=cfg["maximum_close_delay_seconds"]):
                    states[symbol] = {
                        "state": "WAITING_FOR_NEXT_CANDLE_CLOSE",
                        "next_close": str(expected + pd.Timedelta(hours=4)),
                    }
                    continue
                last = con.execute("SELECT MAX(observed) FROM forecasts WHERE symbol=?", (symbol,)).fetchone()[0]
                if last and now - pd.Timestamp(last) < pd.Timedelta(hours=24):
                    states[symbol] = {"state": "OBSERVING_NONOVERLAPPING_24H_WINDOW"}
                    continue
                if symbol not in histories or histories[symbol].index[-1] + pd.Timedelta(hours=4) < expected:
                    histories[symbol], _ = recent_bars(symbol)
                px = histories[symbol]
                close = px.index[-1] + pd.Timedelta(hours=4)
                if close != expected:
                    raise ValueError("Latest completed candle is unavailable; no catch-up forecast")
                values = probabilities(policy, px, cfg)
                entry = quote if injected else next(q for q in quote_snapshot() if q["symbol"] == symbol)
                entry_now = pd.Timestamp(entry["observed_utc"])
                if entry["spread_bps"] > 15 or entry_now - expected > pd.Timedelta(
                    seconds=cfg["maximum_close_delay_seconds"]
                ):
                    raise ValueError("Computation/quote delay missed the entry quality window")
                record(con, symbol, entry, entry_now, px.index[-1], values, cfg)
                states[symbol] = {
                    "state": "NEW_REAL_DATA_PAPER_OBSERVATIONS",
                    "probabilities": values,
                    "orders_enabled": False,
                }
        except Exception as exc:
            states[symbol] = {"state": "ERROR_NO_NEW_FORECAST", "error": str(exc)}
    status = {
        "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
        "exchange_utc": str(now),
        "pid": os.getpid(),
        "state": "paper_observing",
        "assets": states,
        "books": summary(con, cfg),
        "orders_enabled": False,
        "quote_observations": con.execute("SELECT COUNT(*) FROM quotes").fetchone()[0],
        "startup_diagnostic_books": summary(con, cfg, "startup_diagnostics"),
        "startup_diagnostic_note": "A one-time quote-origin forecast starts collection immediately. Its input candle may be up to four hours old, so its timing differs from bar-open training. These diagnostic books are excluded from primary bar-close accuracy and balances.",
        "protocol_sha256": digest(cfg),
        "note": "New real-quote paper books only. Quotes within five minutes of candle close; 24h outcomes. No backfill. Estimated fees and slippage; independent virtual books. Requires this computer awake and online.",
    }
    atomic_json(root / "runtime" / "challengers" / "status.json", status)
    return status


def load_policies(root, cfg):
    folder = root / "runtime" / "challengers"
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(cfg):
        raise ValueError("Frozen challenger protocol mismatch")
    policies = {}
    for symbol, item in manifest["assets"].items():
        if symbol not in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            raise ValueError("No live CFD models supported")
        file = folder / f"{symbol}.joblib"
        if hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Prospective model artifact integrity failure")
        policies[symbol] = joblib.load(file)
    return policies


def run(root: Path = ROOT, once=False):
    cfg = load_challengers(root)
    folder = root / "runtime" / "challengers"
    design = {
        "primary": "New forecasts only within five minutes after completed 4h candles; no catch-up or overlapping 24h targets.",
        "startup_diagnostic": "One quote-origin 24h observation per model/asset at startup; last completed input candle can be up to four hours old. Excluded from primary results.",
        "fills": "Fresh public bid/ask sampled after inference; assumed fees/slippage. Independent books per model, asset and cohort.",
        "label_expiry_seconds": cfg["maximum_label_delay_seconds"],
        "orders_enabled": False,
    }
    design_path = folder / "forward_design.json"
    if design_path.exists():
        if json.loads(design_path.read_text())["sha256"] != digest(design):
            raise ValueError("Forward sampling design changed after inception")
    else:
        atomic_json(
            design_path,
            {"design": design, "sha256": digest(design), "locked_utc": datetime.now(timezone.utc).isoformat()},
        )
    lease = socket.socket()
    lease.bind(("127.0.0.1", 18761))
    lease.listen(1)
    con = database(root)
    policies, histories = load_policies(root, cfg), {}
    try:
        while not (folder / "STOP").exists():
            try:
                observe(con, policies, histories, cfg, root)
            except Exception as exc:
                atomic_json(
                    folder / "status.json",
                    {
                        "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                        "state": "degraded",
                        "pid": os.getpid(),
                        "error": str(exc),
                        "orders_enabled": False,
                    },
                )
                if once:
                    raise
            if once:
                break
            for _ in range(60):
                if (folder / "STOP").exists():
                    break
                time.sleep(1)
        if (folder / "STOP").exists():
            for quote in quote_snapshot():
                with con:
                    mark_outcomes(con, quote["symbol"], quote, pd.Timestamp(quote["observed_utc"]), cfg, stopping=True)
                    mark_outcomes(
                        con,
                        quote["symbol"],
                        quote,
                        pd.Timestamp(quote["observed_utc"]),
                        cfg,
                        stopping=True,
                        table="startup_diagnostics",
                    )
            atomic_json(
                folder / "status.json",
                {
                    "state": "stopped",
                    "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                    "books": summary(con, cfg),
                    "orders_enabled": False,
                },
            )
    finally:
        con.close()
        lease.close()
