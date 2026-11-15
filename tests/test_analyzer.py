"""Analyzer on synthetic daily data (no network): factors are causal, strategies trade next day,
walk-forward ML never trains on its test window, and the signal is well-formed."""
import numpy as np
import pandas as pd
import pytest

from sentinel.analyzer.data import Snapshot, asset_class
from sentinel.analyzer.engine import analyze, combine
from sentinel.analyzer.factors import ALL_FACTORS, compute_factors
from sentinel.analyzer.strategies import positions, strategy_returns


def _px(n=1600, seed=1, drift=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    o = c * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": rng.lognormal(14, 0.3, n)}, index=idx)


def _fetch(sym, period="10y"):
    return _px(seed=abs(hash(sym)) % 1000)


def _snap(sym):
    return Snapshot(name=sym, quote_type="EQUITY")


def test_factors_have_no_lookahead():
    px = _px()
    base = compute_factors(px, {"market": _px(seed=9), "vix": _px(seed=10)})
    alt_px = px.copy()
    alt_px.iloc[1001:] *= 2.0
    alt = compute_factors(alt_px, {"market": _px(seed=9), "vix": _px(seed=10)})
    cols = [c for c in ALL_FACTORS if c in base]
    pd.testing.assert_frame_equal(base.iloc[:1001][cols], alt.iloc[:1001][cols])


def test_strategy_positions_earn_next_day_return():
    px = _px()
    f = compute_factors(px)
    pos = positions(px, f)
    r = strategy_returns(px["close"], pos, cost_bps=0)
    daily = px["close"].pct_change()
    t = px.index[500]
    assert r.at[px.index[501], "sma200"] == pytest.approx(pos.at[t, "sma200"] * daily.iloc[501])


def test_asset_class():
    assert asset_class("BTC-USD", "CRYPTOCURRENCY") == "crypto"
    assert asset_class("GC=F") == "commodity"
    assert asset_class("SPY", "ETF") == "etf"
    assert asset_class("AAPL", "EQUITY") == "stock"


def test_analyze_end_to_end():
    a = analyze("TEST", fetch=_fetch, snapshot=_snap)
    assert a.ml is not None and 0 <= a.ml.current_prob <= 1
    assert a.ml.oos.index.min() > a.prices.index[252 + 756 - 1]
    m = a.ml.metrics
    assert 0.3 < m["auc"] < 0.7            # random walk: no real edge should be found
    assert {"buy_hold", "sma200", "ml", "adaptive"} <= set(a.strategy_table.index)
    assert a.signal["label"] in {"Bullish", "Bearish", "Neutral"} and -1 <= a.signal["score"] <= 1
    assert a.signal["expected_move_1sd"] > 0
    assert len(a.readings) >= 10


def test_ml_without_edge_gets_no_weight():
    table = pd.DataFrame({"strategy": ["Buy & hold"], "family": ["benchmark"], "sharpe": [0.5], "excess_t": [0.0]},
                         index=["buy_hold"])

    class FakeML:
        current_prob = 0.9
        metrics = {"auc": 0.49, "p_value": 0.9, "horizon_days": 5}
    sig = combine(table, {"buy_hold": 1}, FakeML(), [], 0.0, 0, None)
    ml_comp = [c for c in sig["components"] if c["component"] == "Machine learning"][0]
    assert ml_comp["weight"] == 0
    assert sig["quality"] == "Weak"
