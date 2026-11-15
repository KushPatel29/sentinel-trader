"""Restartable, local forward paper worker. Public GET requests only.

Each of 20 predeclared candidates and the validation-selected policy gets an
independent simulated portfolio. No historical/catch-up fills are invented.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import socket
import sqlite3
import time

import joblib
import pandas as pd

from .config import ROOT, atomic_json, digest, load_protocol
from .data import BAR, live_snapshot, read_history, rest_bars, server_now, session, validate
from .execution import Account
from .features import features, rules, volatility_scale
from .models import FAMILIES, THRESHOLDS

log = logging.getLogger("sentinel.paper")


def database(root: Path = ROOT) -> sqlite3.Connection:
    (root / "runtime").mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root / "runtime" / "paper.sqlite", timeout=20)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
          name TEXT NOT NULL, symbol TEXT NOT NULL, state TEXT NOT NULL,
          last_bar TEXT, target REAL NOT NULL DEFAULT 0, PRIMARY KEY(name,symbol));
        CREATE TABLE IF NOT EXISTS fills (
          id TEXT PRIMARY KEY, time TEXT, name TEXT, symbol TEXT, bar TEXT, reason TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS snapshots (
          time TEXT, name TEXT, symbol TEXT, equity REAL, cash REAL, units REAL, price REAL,
          PRIMARY KEY(time,name,symbol));
        CREATE TABLE IF NOT EXISTS events (time TEXT, severity TEXT, symbol TEXT, message TEXT);
        CREATE TABLE IF NOT EXISTS predictions (
          id TEXT PRIMARY KEY, created TEXT, due TEXT, symbol TEXT, family TEXT, bar TEXT,
          probability REAL, entry_mid REAL, exit_mid REAL, actual_up INTEGER, matured TEXT);
    """)
    return con


def event(con, severity, symbol, message):
    con.execute("INSERT INTO events VALUES (?,?,?,?)", (datetime.now(timezone.utc).isoformat(), severity, symbol, message))
    log.log(logging.ERROR if severity == "error" else logging.INFO, "%s %s", symbol, message)


def save_fill(con, fill, name, symbol, bar, now, reason):
    ident = hashlib.sha256(f"{now}|{name}|{symbol}|{reason}".encode()).hexdigest()
    con.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?)", (ident, now, name, symbol, bar, reason, json.dumps(fill)))


def signal_targets(px, policy, cfg):
    f = features(px)
    positions = rules(px, f, cfg)
    targets = {name: float(positions[name].iloc[-1]) for name in positions}
    probs = {}
    scale = float(volatility_scale(f, cfg).iloc[-1])
    for family in FAMILIES:
        p = float(policy["all_models"][family].predict(f.iloc[[-1]])[0])
        probs[family] = p
        for threshold in THRESHOLDS:
            targets[f"{family}@{threshold:.2f}"] = scale if p >= threshold else 0.0
    targets["selected_policy"] = targets[policy["candidate"]]
    return targets, probs


def cycle(con, histories, policies, cfg, root: Path = ROOT) -> dict:
    exchange_now = server_now()
    observed = datetime.now(timezone.utc).isoformat()
    stop = (root / "runtime" / "STOP").exists()
    rows, errors = [], []
    with session() as s, con:
        for symbol in cfg["symbols"]:
            con.execute("SAVEPOINT symbol_cycle")
            try:
                px = histories[symbol]
                recent = rest_bars(symbol, px.index[-1], exchange_now, s=s)
                if not recent.empty:
                    px = pd.concat([px, recent])
                    px = px[~px.index.duplicated(keep="last")].sort_index()
                validate(px, require_contiguous=True)
                histories[symbol] = px
                quote = live_snapshot(symbol, s=s)
                if quote["latency_s"] > 3 or quote["spread_bps"] > 15:
                    raise ValueError("Quote latency or spread exceeds the paper execution limit")
                if exchange_now - (px.index[-1] + BAR) > BAR + pd.Timedelta(minutes=5):
                    raise ValueError("Stale market candles")
                bar = str(px.index[-1])
                saved = con.execute("SELECT name,state,last_bar,target FROM accounts WHERE symbol=?", (symbol,)).fetchall()
                first = len(saved) == 0
                new_bar = first or any(row[2] != bar for row in saved)
                targets, probs = signal_targets(px, policies[symbol], cfg) if new_bar else ({}, {})
                # On first boot we observe the current bar but start cash. Fresh future bars trigger entries.
                signal_fresh = exchange_now - (px.index[-1] + BAR) <= pd.Timedelta(minutes=5)
                if first:
                    for name in targets:
                        account = Account.create(cfg["initial_cash"] / len(cfg["symbols"]))
                        con.execute("INSERT INTO accounts VALUES (?,?,?,?,?)", (name, symbol, json.dumps(account.to_dict()), bar, 0.0))
                    event(con, "info", symbol, "Forward experiment initialized in cash. Waiting for a future completed bar.")
                    saved = con.execute("SELECT name,state,last_bar,target FROM accounts WHERE symbol=?", (symbol,)).fetchall()
                if new_bar and not first:
                    previous = pd.Timestamp(saved[0][2])
                    if px.index[-1] - previous > BAR:
                        event(con, "info", symbol, "Missed candles while offline; skipped historical fills.")
                    if not signal_fresh:
                        event(con, "info", symbol, "Signal arrived over 5 minutes late; flatten and wait for a future bar.")
                    for family, prob in probs.items():
                        if signal_fresh and not stop:
                            # Non-overlapping 24h predictions per family, reducing inflated sample counts.
                            last = con.execute("SELECT created FROM predictions WHERE symbol=? AND family=? ORDER BY created DESC LIMIT 1",
                                               (symbol, family)).fetchone()
                            if last and pd.Timestamp(observed) - pd.Timestamp(last[0]) < pd.Timedelta(hours=24):
                                continue
                            ident = hashlib.sha256(f"{symbol}|{family}|{bar}".encode()).hexdigest()
                            due = str(pd.Timestamp(observed) + pd.Timedelta(hours=24))
                            con.execute("INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                (ident, observed, due, symbol, family, bar, prob, quote["mid"], None, None, None))
                for name, state, last_bar, old_target in saved:
                    account = Account(**json.loads(state))
                    why = account.risk(quote["mid"], observed, cfg)
                    target = old_target
                    if new_bar and not first:
                        target = targets[name] if signal_fresh else 0.0
                    if stop or why:
                        target = 0.0
                    # Only trade on a fresh decision or a risk/stop exit; ticks never rebalance silently.
                    if (new_bar and not first) or stop or why:
                        fill = account.rebalance(target, quote["mid"], cfg, bid=quote["bid"], ask=quote["ask"])
                        if fill:
                            save_fill(con, fill, name, symbol, bar, observed, "stop" if stop else why or "fresh_signal")
                    con.execute("UPDATE accounts SET state=?,last_bar=?,target=? WHERE name=? AND symbol=?",
                                (json.dumps(account.to_dict()), bar, target, name, symbol))
                    con.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?)", (observed, name, symbol,
                                account.equity(quote["mid"]), account.cash, account.units, quote["mid"]))
                matured = con.execute("SELECT id,entry_mid,due FROM predictions WHERE symbol=? AND actual_up IS NULL", (symbol,)).fetchall()
                for ident, entry, due in matured:
                    if pd.Timestamp(due) <= pd.Timestamp(observed):
                        con.execute("UPDATE predictions SET exit_mid=?,actual_up=?,matured=? WHERE id=?",
                                    (quote["mid"], int(quote["mid"] > entry), observed, ident))
                rows.append({"symbol": symbol, "bar": bar, "quote": quote, "new_bar": new_bar,
                             "selected_candidate": policies[symbol]["candidate"], "probabilities": probs})
                con.execute("RELEASE SAVEPOINT symbol_cycle")
            except Exception as exc:
                con.execute("ROLLBACK TO SAVEPOINT symbol_cycle")
                con.execute("RELEASE SAVEPOINT symbol_cycle")
                message = f"{type(exc).__name__}: {exc}"
                errors.append({"symbol": symbol, "error": message})
                event(con, "error", symbol, message)
    # Status never implies a position was closed if a data/quote request failed.
    status = {"heartbeat_utc": observed, "exchange_utc": str(exchange_now), "pid": __import__("os").getpid(),
              "state": "stopped" if stop else "degraded" if errors else "running",
              "mode": "local_forward_paper_only", "protocol_sha256": digest(cfg), "markets": rows,
              "errors": errors, "next_poll_seconds": cfg["poll_seconds"],
              "note": "Local process requires this computer to remain awake and online. No real orders are sent."}
    atomic_json(root / "runtime" / "status.json", status)
    return status


def run(root: Path = ROOT, once=False):
    cfg = load_protocol(root)
    (root / "runtime").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(root / "runtime" / "worker.log", encoding="utf-8")])
    # OS socket lease prevents duplicate workers even after stale PID files or a crash.
    lease = socket.socket()
    lease.bind(("127.0.0.1", 18759))
    lease.listen(1)
    con = database(root)
    try:
        manifest = json.loads((root / "runtime" / "policies" / "manifest.json").read_text())
        if manifest["protocol_sha256"] != digest(cfg):
            raise ValueError("Paper policies do not match the locked protocol")
        histories = {sym: read_history(sym, root) for sym in cfg["symbols"]}
        policies = {sym: joblib.load(root / "runtime" / "policies" / f"{sym}.joblib") for sym in cfg["symbols"]}
        while True:
            try:
                status = cycle(con, histories, policies, cfg, root)
                if once or status["state"] == "stopped":
                    break
            except Exception as exc:
                log.exception("Worker cycle failed")
                con.rollback()
                atomic_json(root / "runtime" / "status.json", {"state": "degraded", "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                            "error": str(exc), "mode": "local_forward_paper_only"})
                if once:
                    raise
            # Sleeps in small pieces so STOP is noticed promptly.
            for _ in range(cfg["poll_seconds"]):
                if (root / "runtime" / "STOP").exists():
                    break
                time.sleep(1)
    finally:
        con.close()
        lease.close()
