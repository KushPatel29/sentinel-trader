"""Prospective challenger forecasts; no trading, no overwrite of v1 paper accounts."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sqlite3
import time

import joblib
import pandas as pd

from .config import ROOT, atomic_json, digest
from .data import live_snapshot, rest_bars, server_now
from .focus_data import load_focus, read_focus, yahoo_context
from .focus_features import build_features, target
from .focus_models import fit_library, fit_model, choose_forecaster
from .focus_research import data_gate, read_context


def prepare(root: Path = ROOT):
    cfg = load_focus(root)
    context = read_context(cfg, root)
    folder = root / "runtime" / "focus"
    folder.mkdir(parents=True, exist_ok=True)
    boundary = pd.Timestamp(cfg["cutoff"])
    current = {}
    for symbol, spec in cfg["assets"].items():
        px = read_focus(symbol, root)
        okay, gate = data_gate(px, spec, boundary-pd.Timedelta(days=cfg["train_days"]+cfg["validation_days"]), boundary)
        if not okay:
            current[symbol] = {"status": "DATA_BLOCKED", "reason": gate}
            continue
        f, groups = build_features(px, spec, context)
        y, end = target(px, spec)
        val_start = boundary-pd.Timedelta(days=cfg["validation_days"])
        models = fit_library(f, y, end, groups, cfg, val_start)
        mask = f.index >= val_start
        probabilities = pd.DataFrame({name: model.predict(f.loc[mask]) for name, model in models.items()}, index=f.index[mask])
        name, scores = choose_forecaster(models, probabilities, y.where(end < boundary), val_start, boundary)
        family, group = name.split(":")
        # Selection is finished; a new prospective fit may now use all pre-inception data.
        current_model = fit_model(family, f, y, end, groups[group], cfg, boundary)
        policy = {"name": name, "model": current_model, "cfg_sha256": digest(cfg), "spec": spec,
                  "effective_after": datetime.now(timezone.utc).isoformat()}
        joblib.dump(policy, folder / f"{symbol}-challenger.joblib")
        scores.to_csv(folder / f"{symbol}-selection.csv", index=False)
        current[symbol] = {"status": "OBSERVATION_ONLY", "model": name, "selection_models_fit_before": str(val_start),
                           "prospective_model_fit_before": str(boundary),
                           "last_training_label_end": current_model.train_last_label_end,
                           "last_calibration_label_end": current_model.calibration_last_label_end,
                           "selected_before": str(boundary), "orders_enabled": False}
    atomic_json(folder / "manifest.json", {"protocol_sha256": digest(cfg), "assets": current,
                "created_utc": datetime.now(timezone.utc).isoformat()})
    return current


def database(root):
    folder = root / "runtime" / "focus"
    folder.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(folder / "forecasts.sqlite", timeout=15)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
      CREATE TABLE IF NOT EXISTS forecasts (
        symbol TEXT, bar TEXT, observed TEXT, due TEXT, model TEXT,
        probability REAL, threshold REAL, entry_price REAL,
        exit_price REAL, actual_up INTEGER, matured TEXT, PRIMARY KEY(symbol,bar));
      CREATE TABLE IF NOT EXISTS events (time TEXT,symbol TEXT,state TEXT,message TEXT);
    """)
    columns = {r[1] for r in con.execute("PRAGMA table_info(forecasts)")}
    if "label_state" not in columns:
        con.execute("ALTER TABLE forecasts ADD COLUMN label_state TEXT NOT NULL DEFAULT 'pending'")
        con.execute("ALTER TABLE forecasts ADD COLUMN label_lateness_seconds REAL")
        con.commit()
    return con


def refresh_context(cfg, root):
    """Future inputs have a separate cache; never overwrite the research snapshots."""
    cache = root / "runtime" / "focus" / "inputs"
    stamp = cache / "refreshed.json"
    date = datetime.now(timezone.utc).date().isoformat()
    if stamp.exists() and json.loads(stamp.read_text()).get("utc_date") == date:
        return
    records = {}
    for key, symbol in cfg["context_symbols"].items():
        _, metadata = yahoo_context(symbol, cfg["start"], datetime.now(timezone.utc).isoformat(), cache)
        records[key] = metadata
    atomic_json(stamp, {"utc_date":date,"context":records})


def observe(con, histories, policies, cfg, root=ROOT):
    now = server_now()
    observed = datetime.now(timezone.utc).isoformat()
    states = {}
    context_root = root / "runtime" / "focus" / "inputs"
    if not (context_root / "refreshed.json").exists():
        context_root = root
    context = read_context(cfg, context_root)
    manifest = json.loads((root / "runtime" / "focus" / "manifest.json").read_text())
    with con:
        for symbol, spec in cfg["assets"].items():
            if symbol not in policies:
                states[symbol] = manifest["assets"][symbol]
                continue
            try:
                px = histories[symbol]
                if spec["source"] == "binance":
                    recent = rest_bars(symbol, px.index[-1], now)
                    if not recent.empty:
                        px = pd.concat([px, recent])
                        px = px[~px.index.duplicated(keep="last")].sort_index()
                    histories[symbol] = px
                duration = pd.Timedelta(spec["interval"])
                last_close = px.index[-1]+duration
                age = (now-last_close).total_seconds()/3600
                if age > (36 if spec["interval"] == "1d" else 6):
                    states[symbol] = {"status": "STALE_REFERENCE_NO_SIGNAL", "last_close": str(last_close), "age_hours": age}
                    continue
                if spec["source"] != "binance":
                    states[symbol] = {"status": "REFERENCE_RESEARCH_NO_LIVE_BROKER_QUOTE", "last_close": str(last_close), "orders_enabled": False}
                    continue
                quote = live_snapshot(symbol)
                if quote["latency_s"] > 3 or quote["spread_bps"] > 15:
                    raise ValueError("Quote quality limit exceeded")
                # Future labels use actual observed mid quotes, not fabricated bar fills.
                for bar, entry, due in con.execute("SELECT bar,entry_price,due FROM forecasts WHERE symbol=? AND actual_up IS NULL AND label_state='pending'", (symbol,)).fetchall():
                    if pd.Timestamp(due) <= now:
                        lateness = (now-pd.Timestamp(due)).total_seconds()
                        if lateness > 300:
                            con.execute("UPDATE forecasts SET label_state='expired_outage',label_lateness_seconds=? WHERE symbol=? AND bar=?",(lateness,symbol,bar))
                            continue
                        exit_price = quote["mid"]
                        con.execute("UPDATE forecasts SET exit_price=?,actual_up=?,matured=?,label_state='observed',label_lateness_seconds=? WHERE symbol=? AND bar=?",
                                    (exit_price, int(exit_price > entry), observed, lateness, symbol, bar))
                policy = policies[symbol]
                f, _ = build_features(px, spec, context)
                probability = float(policy["model"].predict(f.iloc[[-1]])[0])
                states[symbol] = {"status": "OBSERVING_FIXED_CHALLENGER", "model": policy["name"],
                                  "last_close": str(last_close), "up_probability": probability,
                                  "classification_threshold": policy["model"].threshold, "orders_enabled": False}
                # These are quote-to-future-quote forecasts, an explicit approximation
                # to the training bar-open target, and carry no simulated fill/P&L.
                last = con.execute("SELECT observed FROM forecasts WHERE symbol=? ORDER BY observed DESC LIMIT 1", (symbol,)).fetchone()
                if last and pd.Timestamp(observed)-pd.Timestamp(last[0]) < duration*spec["horizon_bars"]:
                    continue
                due = pd.Timestamp(observed)+duration*spec["horizon_bars"]
                con.execute("INSERT OR IGNORE INTO forecasts (symbol,bar,observed,due,model,probability,threshold,entry_price,exit_price,actual_up,matured) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (symbol, str(px.index[-1]), observed,
                            str(due), policy["name"], probability, policy["model"].threshold, quote["mid"], None, None, None))
            except Exception as exc:
                states[symbol] = {"status": "ERROR_NO_SIGNAL", "error": str(exc)}
    status = {"heartbeat_utc": observed, "state": "observing", "pid": __import__("os").getpid(), "assets": states,
              "protocol_sha256": digest(cfg), "orders_enabled": False, "poll_seconds": 60,
              "note": "Prospective quote-to-quote challenger forecasts only. Training targets use bar opens; this execution-timing difference is explicit. No modification of v1 paper accounts."}
    atomic_json(root / "runtime" / "focus" / "status.json", status)
    return status


def run(root: Path = ROOT, once=False):
    cfg = load_focus(root)
    folder = root / "runtime" / "focus"
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol_sha256"] != digest(cfg):
        raise ValueError("Focus policies do not match the protocol")
    lease = socket.socket()
    lease.bind(("127.0.0.1", 18760))
    lease.listen(1)
    policies = {s: joblib.load(folder / f"{s}-challenger.joblib") for s, m in manifest["assets"].items() if m["status"] == "OBSERVATION_ONLY"}
    histories = {s: read_focus(s, root) for s in policies}
    con = database(root)
    try:
        while not (folder / "STOP").exists():
            try:
                refresh_context(cfg, root)
                observe(con, histories, policies, cfg, root)
            except Exception as exc:
                con.rollback()
                atomic_json(folder / "status.json", {"heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                            "state": "degraded", "error": str(exc), "orders_enabled": False})
                if once:
                    raise
            if once:
                break
            for _ in range(60):
                if (folder / "STOP").exists():
                    break
                time.sleep(1)
    finally:
        con.close()
        lease.close()
