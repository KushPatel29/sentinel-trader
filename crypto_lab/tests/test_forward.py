from __future__ import annotations
import json

import pandas as pd
import pytest

from sentinel_crypto.config import load_protocol
from sentinel_crypto.paper import cycle, database


@pytest.fixture
def market(monkeypatch, tmp_path):
    import sentinel_crypto.paper as m
    cfg = {**load_protocol(), "symbols": ["BTCUSDT"]}
    idx = pd.date_range("2026-11-01", periods=15, freq="4h", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.,
                         "volume": 100., "quote_volume": 10000., "trades": 100., "taker_base": 50.}, index=idx)
    state = {"now": idx[-1] + pd.Timedelta(hours=4, seconds=5), "frame": frame, "mid": 100.}
    monkeypatch.setattr(m, "server_now", lambda: state["now"])
    monkeypatch.setattr(m, "rest_bars", lambda *a, **kw: state["frame"])
    monkeypatch.setattr(m, "signal_targets", lambda *a: ({"cash": 0., "selected_policy": .5}, {"logistic": .6}))
    monkeypatch.setattr(m, "live_snapshot", lambda *a, **kw: {"mid": state["mid"], "bid": state["mid"]-.01,
        "ask": state["mid"]+.01, "latency_s": .1, "spread_bps": 2.})
    return m, cfg, state, {"BTCUSDT": frame.copy()}, {"BTCUSDT": {"candidate": "test"}}, tmp_path


def advance(state):
    row = state["frame"].iloc[[-1]].copy()
    row.index = row.index + pd.Timedelta(hours=4)
    state["frame"] = pd.concat([state["frame"], row])
    state["now"] += pd.Timedelta(hours=4)


def test_first_boot_never_backfills_and_repeat_is_idempotent(market):
    m, cfg, state, histories, policies, root = market
    con = database(root)
    assert cycle(con, histories, policies, cfg, root)["state"] == "running"
    assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 0
    advance(state)
    cycle(con, histories, policies, cfg, root)
    assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 1
    cycle(con, histories, policies, cfg, root)
    assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 1
    con.close()
    con = database(root)
    cycle(con, histories, policies, cfg, root)
    assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 1
    con.close()


def test_late_signal_does_not_get_historical_fill(market):
    m, cfg, state, histories, policies, root = market
    con = database(root)
    cycle(con, histories, policies, cfg, root)
    advance(state)
    state["now"] += pd.Timedelta(minutes=10)
    cycle(con, histories, policies, cfg, root)
    assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 0
    con.close()


def test_quote_failure_is_recorded_without_inventing_fill(market, monkeypatch):
    m, cfg, state, histories, policies, root = market
    con = database(root)
    cycle(con, histories, policies, cfg, root)
    def fail(*a, **kw):
        raise TimeoutError("simulated unavailable quote")
    monkeypatch.setattr(m, "live_snapshot", fail)
    advance(state)
    result = cycle(con, histories, policies, cfg, root)
    assert result["state"] == "degraded"
    assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 0
    con.close()


def test_stop_flattens_simulated_position(market):
    m, cfg, state, histories, policies, root = market
    con = database(root)
    cycle(con, histories, policies, cfg, root)
    advance(state)
    cycle(con, histories, policies, cfg, root)
    (root / "runtime" / "STOP").touch()
    result = cycle(con, histories, policies, cfg, root)
    assert result["state"] == "stopped"
    account = json.loads(con.execute("SELECT state FROM accounts WHERE name='selected_policy'").fetchone()[0])
    assert account["units"] == 0
    assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 2
    con.close()


def test_daily_risk_exit_on_quote_tick(market):
    m, cfg, state, histories, policies, root = market
    con = database(root)
    cycle(con, histories, policies, cfg, root)
    advance(state)
    cycle(con, histories, policies, cfg, root)
    state["mid"] = 90.
    cycle(con, histories, policies, cfg, root)
    row = con.execute("SELECT state FROM accounts WHERE name='selected_policy'").fetchone()[0]
    assert json.loads(row)["units"] == 0
    con.close()


def test_partial_symbol_cycle_rolls_back_fills_and_state(market, monkeypatch):
    m, cfg, state, histories, policies, root = market
    con = database(root)
    cycle(con, histories, policies, cfg, root)
    old_bar = con.execute("SELECT last_bar FROM accounts LIMIT 1").fetchone()[0]
    advance(state)
    original = m.save_fill
    def fail_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Crash after fill insertion, before account update")
    monkeypatch.setattr(m, "save_fill", fail_after_insert)
    result = cycle(con, histories, policies, cfg, root)
    assert result["state"] == "degraded"
    assert con.execute("SELECT count(*) FROM fills").fetchone()[0] == 0
    assert con.execute("SELECT last_bar FROM accounts LIMIT 1").fetchone()[0] == old_bar
    con.close()
