"""Daily gold plan executor against a fake Alpaca client."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sentinel.gold.plan_bot as B


def _plan(action, entry=None, stop=None):
    return SimpleNamespace(action=action, headline=action.lower(), entry=entry, stop=stop, trigger_above=None)


def test_decide_sizes_entry_by_risk():
    i = B.decide(_plan("ENTER LONG", 300.0, 290.0), 0, None, equity=100_000, buying_power=100_000, risk=0.01)
    assert i.kind == "buy_oto" and i.qty == 100 and i.stop == 290.0          # $1,000 risk / $10 per share
    capped = B.decide(_plan("ENTER LONG", 300.0, 299.5), 0, None, 100_000, 30_000, 0.01)
    assert capped.qty == 98                                                  # buying power cap (no margin)


def test_decide_trails_but_never_lowers_stop():
    assert B.decide(_plan("HOLD LONG", 300, 295.0), 50, None, 1e5, 1e5).kind == "place_stop"
    assert B.decide(_plan("HOLD LONG", 300, 296.0), 50, 295.0, 1e5, 1e5).kind == "raise_stop"
    assert B.decide(_plan("HOLD LONG", 300, 294.0), 50, 295.0, 1e5, 1e5).kind == "none"


def test_decide_exits_and_stands_aside():
    assert B.decide(_plan("EXIT"), 50, 295.0, 1e5, 1e5).kind == "sell"
    assert B.decide(_plan("STAND ASIDE"), 0, None, 1e5, 1e5).kind == "none"


class FakeClient:
    def __init__(self, qty=0, stop=None):
        self.calls, self.qty, self.stop = [], qty, stop

    def get_account(self):
        return SimpleNamespace(equity="100000", buying_power="100000")

    def get_all_positions(self):
        return [SimpleNamespace(symbol="GLD", qty=str(self.qty))] if self.qty else []

    def get_orders(self, filter):
        if self.stop is None:
            return []
        return [SimpleNamespace(id="s1", order_type=SimpleNamespace(value="stop"), side=SimpleNamespace(value="sell"),
                                stop_price=str(self.stop))]

    def submit_order(self, req):
        self.calls.append(("submit", req))

    def replace_order_by_id(self, oid, req):
        self.calls.append(("replace", oid, req))

    def cancel_order_by_id(self, oid):
        self.calls.append(("cancel", oid))


def _fake_fetch(sym):
    idx = pd.bdate_range("2020-01-01", periods=900)
    c = pd.Series(100 * np.exp(np.linspace(0, 0.5, 900)), index=idx)
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1e6})


@pytest.mark.parametrize("action,qty,stop,expected", [
    ("ENTER LONG", 0, None, "submit"), ("HOLD LONG", 100, None, "submit"), ("HOLD LONG", 100, 90.0, "replace"),
    ("EXIT", 100, 90.0, "cancel")])
def test_run_sends_the_right_orders(monkeypatch, action, qty, stop, expected):
    monkeypatch.setattr(B, "todays_plan", lambda g, c, financing=0.0: _plan(action, 150.0, 140.0))
    cl = FakeClient(qty, stop)
    _, intent = B.run(cl, fetch=_fake_fetch, now=pd.Timestamp("2026-11-21 12:00", tz="UTC"))
    assert cl.calls and cl.calls[0][0] == expected
    if action == "ENTER LONG":
        req = cl.calls[0][1]
        assert req.order_class.value == "oto" and req.stop_loss.stop_price == 140.0 and req.qty == 100


def test_dry_run_sends_nothing(monkeypatch):
    monkeypatch.setattr(B, "todays_plan", lambda g, c, financing=0.0: _plan("ENTER LONG", 150.0, 140.0))
    cl = FakeClient()
    B.run(cl, dry_run=True, fetch=_fake_fetch, now=pd.Timestamp("2026-11-21 12:00", tz="UTC"))
    assert cl.calls == []


def test_gld_unfinished_session_dropped():
    last_day = _fake_fetch("GLD").index[-1].date().isoformat()
    g, _ = B.load_gld(_fake_fetch, now=pd.Timestamp(last_day + " 18:00", tz="UTC"))
    assert g.index[-1] < _fake_fetch("GLD").index[-1]          # 13:00 New York: GLD still trading
