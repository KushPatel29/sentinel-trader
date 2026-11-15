from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel_crypto.config import load_protocol, atomic_json, digest
from sentinel_crypto.data import normalize, validate, COLUMNS, valid_symbol
from sentinel_crypto.execution import Account, simulate
from sentinel_crypto.features import features, labels, rules
from sentinel_crypto.metrics import performance, combine, bootstrap_mean
from sentinel_crypto.models import fit


@pytest.fixture
def cfg():
    return {**load_protocol(), "warmup_bars": 1200, "daily_loss_limit": 0.5, "drawdown_limit": 0.9}


@pytest.fixture
def px():
    rng = np.random.default_rng(19)
    n = 5600
    c = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.014, n)))
    o = np.r_[100, c[:-1]]
    volume = rng.uniform(500, 2000, n)
    return pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) * 1.005,
        "low": np.minimum(o, c) * 0.995, "volume": volume, "quote_volume": volume*c,
        "trades": np.full(n, 1000), "taker_base": volume*rng.uniform(.3, .7, n)},
        index=pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC", name="timestamp"))


def test_features_do_not_change_when_future_prices_change(px):
    cut = 4000
    expected = features(px.iloc[:cut])
    changed = px.copy()
    changed.iloc[cut:, :] *= 5
    pd.testing.assert_frame_equal(features(changed).iloc[:cut], expected)


def test_rule_decisions_are_causal(px, cfg):
    cut = 3900
    pd.testing.assert_frame_equal(rules(px, features(px), cfg).iloc[:cut],
                                 rules(px.iloc[:cut], features(px.iloc[:cut]), cfg))


def test_ml_training_does_not_see_future_outcomes(px, cfg):
    f = features(px)
    boundary = px.index[4500]
    y = labels(px, cfg["horizon_bars"])
    a = fit("logistic", f, y, cfg, boundary)
    changed = y.copy()
    changed.loc[boundary - pd.Timedelta(hours=24):] = 100
    b = fit("logistic", f, changed, cfg, boundary)
    np.testing.assert_allclose(a.predict(f.iloc[4500:4550]), b.predict(f.iloc[4500:4550]), atol=1e-12)
    last_label_available = pd.Timestamp(a.calibration_last) + pd.Timedelta(hours=28)
    assert last_label_available < boundary
    assert pd.Timestamp(a.train_last) < pd.Timestamp(a.calibration_last)


def test_labels_use_next_open_and_leave_unknown_tail(px):
    y = labels(px, 6)
    assert y.iloc[20] == pytest.approx(px.open.iloc[27] / px.open.iloc[21] - 1)
    assert y.iloc[-7:].isna().all()


def test_completed_candles_only_and_both_timestamp_units():
    t = int(pd.Timestamp("2026-11-15T00:00:00Z").timestamp() * 1000)
    rows = [[t, 100, 110, 90, 105, 20, t+1, 2100, 30, 11, 1200, 0],
            [(t+14400000)*1000, 105, 110, 90, 101, 20, 0, 2000, 30, 11, 1200, 0]]
    df = normalize(pd.DataFrame(rows, columns=COLUMNS), pd.Timestamp("2026-11-15T05:00:00Z"))
    assert len(df) == 1
    assert df.index[0] == pd.Timestamp("2026-11-15T00:00:00Z")


def test_gaps_are_rejected(px):
    with pytest.raises(ValueError, match="gaps"):
        validate(px.drop(px.index[10]), require_contiguous=True)


@pytest.mark.parametrize("symbol", ["BTCUSDT/../../bad", "BTC-USD", "btc usdt", "<script>"])
def test_symbol_validation(symbol):
    with pytest.raises(ValueError):
        valid_symbol(symbol)


def test_ohlc_bounds_rejected(px):
    px.iloc[10, px.columns.get_loc("high")] = 0
    with pytest.raises(ValueError):
        validate(px)


def test_decision_cannot_earn_same_candle_return(cfg):
    p = pd.DataFrame({"open": [100, 200, 200], "close": [200, 200, 200]},
                     index=pd.date_range("2026-01-01", periods=3, freq="4h", tz="UTC"))
    path, fills, _ = simulate(p, pd.Series([.6, 0, 0], index=p.index), cfg)
    assert path.equity.iloc[-1] < 1
    assert fills[0]["timestamp"] == str(p.index[1])
    assert fills[0]["price"] > 200


def test_terminal_exit_and_fees_are_included(cfg):
    p = pd.DataFrame({"open": [100]*4, "close": [100]*4},
                     index=pd.date_range("2026-01-01", periods=4, freq="4h", tz="UTC"))
    target = pd.Series(.6, index=p.index)
    base, fills, acc = simulate(p, target, cfg)
    stress, _, _ = simulate(p, target, cfg, multiplier=2)
    assert base.equity.iloc[-1] < 1
    assert stress.equity.iloc[-1] < base.equity.iloc[-1]
    assert fills[-1]["reason"] == "end_of_test"
    assert acc.closed == 1 and acc.units == 0 and acc.fees > 0


def test_position_cap_and_cash_conservation(cfg):
    acc = Account.create(10000)
    acc.rebalance(10, 100, cfg)
    assert acc.cash >= 0
    assert acc.units * 100 / acc.equity(100) < .601
    acc.rebalance(0, 100, cfg)
    assert acc.units == 0
    assert acc.cash < 10000


def test_daily_halt_and_new_day(cfg):
    cfg = {**cfg, "daily_loss_limit": .02}
    acc = Account.create(10000)
    acc.risk(100, "2026-11-15", cfg)
    acc.rebalance(.6, 100, cfg)
    assert acc.risk(90, "2026-11-15", cfg) == "daily_halt"
    acc.rebalance(0, 90, cfg)
    assert acc.risk(90, "2026-11-15", cfg) == "daily_halt"
    assert acc.risk(90, "2026-11-16", cfg) is None


def test_max_drawdown_halt_persists(cfg):
    cfg = {**cfg, "drawdown_limit": .05}
    acc = Account.create(10000)
    acc.rebalance(.6, 100, cfg)
    assert acc.risk(80, "2026-11-15", cfg) == "drawdown_halt"
    assert acc.risk(100, "2026-11-16", cfg) == "drawdown_halt"


def test_drawdown_includes_initial_capital():
    p = pd.DataFrame({"equity": [.9, .95], "return": [-.1, .95/.9-1], "exposure": [0, 0]},
                     index=pd.date_range("2026-01-01", periods=2, freq="4h", tz="UTC"))
    assert performance(p)["max_drawdown"] == pytest.approx(-.1)


def test_portfolio_uses_fixed_sleeves():
    idx = pd.date_range("2026-01-01", periods=2, freq="4h", tz="UTC")
    def path(v):
        return pd.DataFrame({"equity": v, "return": [0, 0], "exposure": [1, 1]}, index=idx)
    p = combine({"a": path([2, 4]), "b": path([1, .5])})
    assert p.equity.iloc[-1] == 2.25
    assert p["return"].iloc[1] == .5


def test_cash_bootstrap_is_not_evidence_of_positive_return():
    daily = pd.Series(0., index=pd.date_range("2026-01-01", periods=180))
    m = bootstrap_mean(daily)
    assert m["ci95_daily_mean"] == [0, 0]
    assert m["prob_mean_positive"] == 0


def test_report_cannot_be_reused_after_protocol_change(tmp_path):
    from sentinel_crypto.research import evaluate
    cfg = load_protocol()
    atomic_json(tmp_path / "protocol.json", cfg)
    atomic_json(tmp_path / "reports" / cfg["version"] / "summary.json", {"protocol_sha256": digest(cfg)})
    cfg["fee_bps"] = 0
    atomic_json(tmp_path / "protocol.json", cfg)
    with pytest.raises(ValueError, match="Sealed protocol changed"):
        evaluate(tmp_path)
