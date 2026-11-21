"""Gold module on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from sentinel.gold import model as M
from sentinel.gold.data import CONTEXT


def _universe(n=4200, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    vol = 0.01 * np.exp(np.cumsum(rng.normal(0, 0.03, n)) * 0.2)          # clustered volatility
    r = rng.standard_t(5, n) * vol / np.sqrt(5 / 3)
    c = 1000 * np.exp(np.cumsum(r))
    hi = c * np.exp(np.abs(rng.normal(0, 1, n)) * vol * 0.6)
    lo = c * np.exp(-np.abs(rng.normal(0, 1, n)) * vol * 0.6)
    gold = pd.DataFrame({"open": c, "high": np.maximum(hi, c), "low": np.minimum(lo, c), "close": c,
                         "volume": 1e5}, index=idx)
    ctx = pd.DataFrame({k: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for k in CONTEXT}, index=idx)
    ctx["gvz"] = 15 + 5 * rng.random(n)
    ctx["gld_volume"] = rng.lognormal(15, 0.3, n)
    return gold, ctx


@pytest.fixture(scope="module")
def uni():
    return _universe()


def test_ranges_are_calibrated_and_ordered(uni):
    gold, ctx = uni
    R = M.ranges(gold, ctx)
    last = gold["close"].iloc[-1]
    for _, row in R.bands.iterrows():
        assert row.low < last < row.high
        assert abs(row.measured_coverage - row.level) < 0.08      # causal rolling quantiles stay near nominal
    w = R.bands.assign(w=R.bands.high - R.bands.low).set_index(["days", "level"])["w"]
    assert w[(1, 0.5)] < w[(1, 0.8)] < w[(1, 0.9)] < w[(21, 0.9)]
    hl = R.highlow.set_index(["side", "confidence"])["level"]
    assert hl[("high", 0.9)] > hl[("high", 0.5)] > last > hl[("low", 0.5)] > hl[("low", 0.9)]


def test_direction_on_random_walk_finds_no_edge(uni):
    gold, ctx = uni
    d = M.direction(gold, ctx)
    assert set(d["days"]) == {5, 21}
    assert d["prob_up"].between(0, 1).all()
    assert not d["reliable"].any()


def test_strategies_and_helpers(uni):
    gold, ctx = uni
    S, curves = M.strategies(gold, start="2015-01-01")
    assert "Buy & hold" in set(S["strategy"]) and len(curves.columns) == len(S)
    assert len(M.seasonality(gold)) == 12
    assert {"driver", "pushing gold"} <= set(M.drivers(gold, ctx).columns)
    s = M.sentiment(gold, ctx)
    assert 0 <= s["greed_index"] <= 100


def test_position_size():
    p = M.position_size(10_000, 0.01, entry=2000, stop=1980, oz_per_lot=100)
    assert p["ounces"] == pytest.approx(5.0) and p["lots"] == pytest.approx(0.05)
    assert M.position_size(10_000, 0.01, 2000, 2000)["ounces"] == 0


def test_financing_lowers_long_returns(uni):
    gold, _ = uni
    a, _ = M.strategies(gold, start="2015-01-01")
    b, _ = M.strategies(gold, start="2015-01-01", financing_annual=0.06)
    ra = a.set_index("strategy")["annual_return"]
    rb = b.set_index("strategy")["annual_return"]
    assert rb["Buy & hold"] < ra["Buy & hold"] - 0.04


def test_extended_drivers_offline(tmp_path, monkeypatch):
    import json

    from sentinel.gold import extra
    cot = [{"report_date_as_yyyy_mm_dd": f"2026-0{m}-0{d}T00:00:00.000", "m_money_positions_long_all": str(100 + m * d),
            "m_money_positions_short_all": "20", "open_interest_all": "400"} for m in range(1, 10) for d in (1, 8)]
    (tmp_path / "cot.json").write_text(json.dumps(cot))
    rows = "\n".join(f"{1_700_000_000 + i * 86400},1,1,1,{1 + i / 100},{1 + i / 100},0" for i in range(400))
    (tmp_path / "extra.json").write_text(json.dumps({"daily": {s: rows for s in ("HYG", "LQD", "^IRX", "INR=X", "CNY=X",
                                                                                  "BTC-USD")}}))
    monkeypatch.setenv("SENTINEL_OFFLINE_COT", str(tmp_path / "cot.json"))
    monkeypatch.setenv("SENTINEL_OFFLINE_GOLD_EXTRA", str(tmp_path / "extra.json"))
    monkeypatch.setenv("SENTINEL_DISABLE_LIVE", "1")
    t = extra.extended_drivers()
    names = set(t["driver"])
    assert "Hedge-fund net position (CFTC)" in names and "Bitcoin" in names
    assert not any("real yield" in n for n in names)          # FRED offline -> silently skipped
    assert t["3-year percentile"].between(0, 1).all()


def test_daily_plan_is_consistent(uni):
    from sentinel.gold.plan import todays_plan
    gold, ctx = uni
    p = todays_plan(gold, ctx)
    assert p.action in {"ENTER LONG", "HOLD LONG", "EXIT", "STAND ASIDE"}
    T = p.trades
    assert len(T) > 10
    assert (pd.to_datetime(T["exit_date"]) >= pd.to_datetime(T["entry_date"])).all()
    stop_exits = T[T["reason"] == "trailing stop"]
    assert (stop_exits["exit"] <= stop_exits["stop_at_exit"] + 1e-9).all()       # never filled better than the stop
    if p.action in ("ENTER LONG", "HOLD LONG"):
        assert p.stop < gold["close"].iloc[-1]
    else:
        assert p.trigger_above == pytest.approx(max(p.sma50, p.sma200))


def test_unfinished_session_is_not_used_for_signals():
    from sentinel.gold.data import split_provisional
    g = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-11-19", "2026-11-20"]))
    during, live = split_provisional(g, pd.Timestamp("2026-11-20 15:00", tz="UTC"))     # 10:00 New York
    assert live["provisional"] and len(during) == 1 and live["price"] == 2.0
    after, live = split_provisional(g, pd.Timestamp("2026-11-20 22:30", tz="UTC"))      # 17:30 New York
    assert not live["provisional"] and len(after) == 2
    next_session, live = split_provisional(g, pd.Timestamp("2026-11-19 23:30", tz="UTC"))  # Thu 18:30, Friday bar open
    assert live["provisional"] and len(next_session) == 1


def test_plan_reports_one_percent_risk_results(uni):
    from sentinel.gold.plan import todays_plan
    gold, ctx = uni
    st_ = todays_plan(gold, ctx).stats
    assert {"account_return_1pct_risk", "account_drawdown_1pct_risk"} <= set(st_.columns)
    assert (st_["account_drawdown_1pct_risk"] <= 0).all()
