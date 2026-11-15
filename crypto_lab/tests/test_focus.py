import zipfile

import numpy as np
import pandas as pd
import pytest

from sentinel_crypto.focus_data import import_broker_csv, load_focus, parse_reference
from sentinel_crypto.focus_execution import simulate_focus
from sentinel_crypto.focus_features import build_features, signal_series, target
from sentinel_crypto.focus_models import fit_model
from sentinel_crypto.focus_research import data_gate, nonoverlapping


@pytest.fixture
def focus_cfg():
    return load_focus()


@pytest.fixture
def hourly():
    idx = pd.date_range("2018-01-01", periods=28000, freq="1h", tz="UTC")
    rng = np.random.default_rng(31)
    c = 1300*np.exp(np.cumsum(rng.normal(0, .001, len(idx))))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c)*1.001, "low": np.minimum(o,c)*.999,
                         "close": c, "minute_count": 60}, index=idx)


def test_fixed_est_conversion_does_not_apply_daylight_savings(tmp_path):
    p = tmp_path / "prices.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("prices.csv", "20250701 080000;100;101;99;100.5;0\n20250701 080100;100.5;102;100;101;0\n")
    out = parse_reference(p, pd.Timestamp("2025-07-02", tz="UTC"))
    assert out.index[0] == pd.Timestamp("2025-07-01T13:00:00Z")
    assert out.minute_count.iloc[0] == 2
    assert out.open.iloc[0] == 100 and out.close.iloc[0] == 101


def test_reference_parser_handles_chunk_boundary_ohlc(tmp_path):
    p = tmp_path / "prices.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("first.csv", "20250101 080000;100;102;99;101;0\n")
        z.writestr("second.csv", "20250101 080100;101;103;100;102;0\n")
    out = parse_reference(p, pd.Timestamp("2025-01-02", tz="UTC"))
    assert len(out) == 1 and out.minute_count.iloc[0] == 2
    assert out.high.iloc[0] == 103


def test_focus_features_and_context_are_causal(hourly, focus_cfg):
    spec = focus_cfg["assets"]["XAUUSD"]
    daily = pd.DataFrame({"close": np.linspace(20, 50, 1200)}, index=pd.date_range("2018-01-01", periods=1200, freq="1D", tz="UTC"))
    cut = 22000
    before, groups = build_features(hourly, spec, {"dollar": daily})
    changed = hourly.copy()
    changed.iloc[cut:] *= 2
    daily2 = daily.copy()
    daily2.loc[daily2.index >= hourly.index[cut].normalize(), "close"] *= 3
    after, _ = build_features(changed, spec, {"dollar": daily2})
    pd.testing.assert_frame_equal(before.iloc[:cut], after.iloc[:cut])
    assert not any("flow_" in c for c in before)
    assert all(not c.startswith("context_") for c in groups["price"])


def test_context_available_only_after_next_midnight(hourly, focus_cfg):
    daily = pd.DataFrame({"close": [10., 20.]}, index=pd.date_range("2018-01-01", periods=2, freq="1D", tz="UTC"))
    f, _ = build_features(hourly.iloc[:72], focus_cfg["assets"]["XAUUSD"], {"dollar": daily})
    assert f.loc["2018-01-02T23:00:00Z", "context_dollar_ret1"] != 1.
    assert f.loc["2018-01-03T00:00:00Z", "context_dollar_ret1"] == 1.


def test_cfd_labels_exclude_rollover_and_weekend_windows(hourly, focus_cfg):
    spec = focus_cfg["assets"]["XAUUSD"]
    y, ends = target(hourly, spec)
    assert y.loc["2018-01-02T07:00:00Z"] == pytest.approx(hourly.loc["2018-01-02T12:00:00Z", "open"]/hourly.loc["2018-01-02T08:00:00Z", "open"]-1)
    assert pd.isna(y.loc["2018-01-02T19:00:00Z"])


def test_calibration_and_label_end_purged_before_validation(hourly, focus_cfg):
    spec = focus_cfg["assets"]["XAUUSD"]
    f, groups = build_features(hourly, spec)
    y, end = target(hourly, spec)
    boundary = pd.Timestamp("2020-08-01", tz="UTC")
    a = fit_model("logistic", f, y, end, groups["price"], focus_cfg, boundary)
    altered = y.copy()
    altered.loc[end >= boundary] = 100
    b = fit_model("logistic", f, altered, end, groups["price"], focus_cfg, boundary)
    assert pd.Timestamp(a.calibration_last_label_end) < boundary
    assert pd.Timestamp(a.train_last_label_end) < boundary-pd.Timedelta(days=120)
    np.testing.assert_allclose(a.predict(f.iloc[-50:]), b.predict(f.iloc[-50:]))


def simple_prices():
    idx = pd.date_range("2026-01-05T07:00:00Z", periods=15, freq="1h")
    return pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "minute_count": 60}, index=idx)


def test_cfd_spread_and_terminal_costs(focus_cfg):
    p = simple_prices()
    decisions = pd.Series(.6, index=p.index)
    spec = focus_cfg["assets"]["XAUUSD"]
    base, fills, stats = simulate_focus(p, decisions, focus_cfg, spec, p.index[0], p.index[-1]+pd.Timedelta(hours=1))
    stress, _, _ = simulate_focus(p, decisions, focus_cfg, spec, p.index[0], p.index[-1]+pd.Timedelta(hours=1), multiplier=4)
    assert stress.equity.iloc[-1] < base.equity.iloc[-1] < 1
    assert stats["overnight_cfd_carries"] == 0
    assert any(f["reason"] == "session_flat" for f in fills)
    assert base.exposure.iloc[-1] == 0


def test_cfd_short_can_profit_from_a_fall_without_leverage(focus_cfg):
    p = simple_prices()
    p.loc[p.index >= p.index[2], ["open", "close"]] = 90
    spec = focus_cfg["assets"]["XAUUSD"]
    no_risk = {**focus_cfg, "daily_loss_limit": .99, "drawdown_limit": .99}
    decisions = pd.Series(-.6, index=p.index)
    path, _, _ = simulate_focus(p, decisions, no_risk, spec, p.index[0], p.index[-1]+pd.Timedelta(hours=1))
    assert path.equity.iloc[-1] > 1
    assert path.exposure.max() < .61


def test_flip_closes_old_trade_and_opens_new_trade(focus_cfg):
    p = simple_prices()
    decisions = pd.Series([.6]*4+[-.6]*11, index=p.index)
    _, _, stats = simulate_focus(p, decisions, focus_cfg, focus_cfg["assets"]["XAUUSD"], p.index[0], p.index[-1]+pd.Timedelta(hours=1))
    assert stats["round_trips"] == 2
    assert stats["entries"] == 2


def test_crypto_spot_does_not_execute_short(focus_cfg):
    p = simple_prices()
    decisions = pd.Series(-.6, index=p.index)
    path, fills, _ = simulate_focus(p, decisions, focus_cfg, focus_cfg["assets"]["BTCUSDT"], p.index[0], p.index[-1]+pd.Timedelta(hours=1))
    assert not fills and path.equity.iloc[-1] == 1


def test_data_gate_rejects_large_history_gap(hourly, focus_cfg):
    broken = hourly[(hourly.index < "2019-01-01")|(hourly.index > "2020-01-01")]
    okay, report = data_gate(broken, focus_cfg["assets"]["XAUUSD"], hourly.index[0], hourly.index[-1])
    assert not okay and report["max_gap_days"] > 300


def test_nonoverlapping_labels_are_not_counted_as_independent_every_hour():
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
    frame = pd.DataFrame({"timestamp": idx, "label_end": idx+pd.Timedelta(hours=5)})
    assert len(nonoverlapping(frame)) == 5


def test_broker_import_requires_explicit_timezones_and_spreads(tmp_path):
    path = tmp_path / "broker.csv"
    path.write_text("timestamp,bid_open,bid_high,bid_low,bid_close,ask_open,ask_close\n2026-01-01 10:00,100,101,99,100,100.1,100.1\n")
    with pytest.raises(ValueError, match="UTC offset"):
        import_broker_csv(path, "XAUUSD", tmp_path)
    path.write_text("timestamp,bid_open,bid_high,bid_low,bid_close,ask_open,ask_close\n2026-01-01T10:00:00Z,100,101,99,100,99,100.1\n")
    with pytest.raises(ValueError, match="Ask below bid"):
        import_broker_csv(path, "XAUUSD", tmp_path)


def test_hysteresis_signal_is_causal():
    idx = pd.date_range("2026-01-01", periods=12, freq="1h")
    p = pd.Series([.6,.4,.6,.4,.6,.4,.6,.4,.6,.4,.6,.4], index=idx)
    full = signal_series(p, .57, "long_short", 4)
    pd.testing.assert_series_equal(full.iloc[:8], signal_series(p.iloc[:8], .57, "long_short", 4))
    assert full.iloc[1] == 1


def test_daily_oil_reference_has_valid_feature_windows(focus_cfg):
    idx = pd.date_range("2021-01-01", periods=1000, freq="1D", tz="UTC")
    p = pd.DataFrame({"open": 80., "high": 81., "low": 79., "close": 80.}, index=idx)
    f, groups = build_features(p, focus_cfg["assets"]["WTI_REFERENCE"])
    assert "ma_distance_5d" in f and len(groups["price"]) > 20


def test_all_missing_features_are_removed_using_training_only(hourly, focus_cfg):
    f, groups = build_features(hourly, focus_cfg["assets"]["XAUUSD"])
    f["never_observed"] = np.nan
    y, end = target(hourly, focus_cfg["assets"]["XAUUSD"])
    m = fit_model("hist_small", f, y, end, groups["price"]+["never_observed"], focus_cfg, pd.Timestamp("2020-08-01", tz="UTC"))
    assert "never_observed" not in m.columns
    assert np.isfinite(m.predict(f.iloc[-5:])).all()
