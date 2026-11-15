import joblib
import numpy as np
import pandas as pd
import pytest

from sentinel_crypto.challenger_models import fit_challenger, hybrid_weights, sequence_batch, split_positions
from sentinel_crypto.challenger_research import load_challengers, evaluation_positions
from sentinel_crypto.challenger_watch import database, mark_outcomes, observe, summary


def fixture_data():
    index = pd.date_range("2024-01-01", periods=1500, freq="4h", tz="UTC")
    rng = np.random.default_rng(44)
    f = pd.DataFrame(
        rng.normal(size=(1500, 7)),
        index=index,
        columns=["ret_1", "vol_1d", "atr", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"],
    )
    y = pd.Series(rng.choice([-0.01, 0.01], size=1500), index=index)
    ends = pd.Series(index + pd.Timedelta(hours=28), index=index)
    return f, y, ends, index[1200]


def test_split_purges_outcomes_and_keeps_calibration_separate():
    f, y, ends, boundary = fixture_data()
    cfg = load_challengers()
    train, stop, cal = split_positions(f, y, ends, boundary, cfg)
    cal_start = boundary - pd.Timedelta(days=60)
    stop_start = cal_start - pd.Timedelta(days=45)
    assert ends.iloc[train].max() < stop_start
    assert f.index[stop].min() >= stop_start and ends.iloc[stop].max() < cal_start
    assert f.index[cal].min() >= cal_start and ends.iloc[cal].max() < boundary


def test_future_features_and_labels_cannot_change_earlier_fit():
    f, y, ends, boundary = fixture_data()
    cfg = load_challengers()
    first = fit_challenger("logistic", f, y, ends, boundary, cfg)
    changed = f.copy()
    changed.loc[changed.index >= boundary, "ret_1"] = 1e9
    altered = y.copy()
    altered.loc[ends >= boundary] *= -1
    second = fit_challenger("logistic", changed, altered, ends, boundary, cfg)
    np.testing.assert_allclose(first.predict(f, [1100]), second.predict(changed, [1100]), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(first.scaler.mean_, second.scaler.mean_)


def test_sequences_end_at_current_decision_and_never_read_future():
    values = np.arange(20)[:, None]
    np.testing.assert_array_equal(sequence_batch(values, [5, 8], 3)[:, :, 0], [[3, 4, 5], [6, 7, 8]])


def test_nonoverlap_uses_actual_target_end_and_preserves_reference_grid():
    f, y, ends, _ = fixture_data()
    spec = {"interval": "4h"}
    ids = evaluation_positions(f, y, ends, f.index[100], f.index[200], spec, reference=f.index[100:200:6])
    assert len(ids) > 5
    assert all(f.index[b] + pd.Timedelta(hours=4) >= ends.iloc[a] for a, b in zip(ids[:-1], ids[1:]))


def test_hybrid_is_a_regularized_convex_mixture():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.3, 0.7, (200, 4))
    y = rng.integers(0, 2, 200)
    w = hybrid_weights(p, y, load_challengers())
    assert np.all(w >= 0) and np.isclose(w.sum(), 1)
    assert np.all((p @ w >= p.min(axis=1)) & (p @ w <= p.max(axis=1)))


def add_long(con, due):
    con.execute(
        "INSERT INTO forecasts(symbol,model,bar,observed,due,probability,entry_mid,action,notional,entry_fill,entry_fee) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        ("BTCUSDT", "lstm", "bar", "2026-11-16T00:00:00Z", due, 0.6, 100.0, "long", 1000.0, 100.1, 1.0),
    )
    con.commit()


def test_actual_direction_can_be_correct_while_paper_trade_loses_after_costs(tmp_path):
    con = database(tmp_path)
    cfg = load_challengers()
    add_long(con, "2026-11-17T00:00:00Z")
    q = {"bid": 100.01, "mid": 100.02, "ask": 100.03}
    with con:
        mark_outcomes(con, "BTCUSDT", q, pd.Timestamp("2026-11-17T00:00:30Z"), cfg)
    row = con.execute("SELECT actual_up,label_state,paper_pnl FROM forecasts").fetchone()
    assert row[0] == 1 and row[1] == "observed" and row[2] < 0
    assert summary(con, cfg)[0]["observed_outcomes"] == 1
    con.close()


def test_missed_outcome_expires_without_inventing_an_on_time_fill(tmp_path):
    con = database(tmp_path)
    cfg = load_challengers()
    add_long(con, "2026-11-17T00:00:00Z")
    with con:
        mark_outcomes(con, "BTCUSDT", {"bid": 90.0, "mid": 90.01}, pd.Timestamp("2026-11-18T00:00:00Z"), cfg)
    row = con.execute("SELECT actual_up,label_state,exit_fill,lateness_seconds FROM forecasts").fetchone()
    assert row[0] is None and row[1] == "expired_outage" and row[2] < 90 and row[3] == 86400
    assert summary(con, cfg)[0]["observed_outcomes"] == 0
    con.close()


def test_start_between_closes_collects_real_quotes_without_backfill(tmp_path, monkeypatch):
    cfg = load_challengers()
    con = database(tmp_path)
    q = {
        "symbol": "BTCUSDT",
        "observed_utc": "2026-11-16T13:30:00Z",
        "bid": 100.0,
        "ask": 100.01,
        "mid": 100.005,
        "spread_bps": 1.0,
    }
    import sentinel_crypto.challenger_watch as watch

    monkeypatch.setattr(watch, "recent_bars", lambda s: pytest.fail("No catch-up candle query expected"))
    result = observe(
        con, {"BTCUSDT": {"boundary": "2026-11-16T12:00:00Z"}}, {}, cfg, tmp_path, quotes=[q], include_startup=False
    )
    assert result["assets"]["BTCUSDT"]["state"] == "WAITING_FOR_NEXT_CANDLE_CLOSE"
    assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 0
    assert result["quote_observations"] == 1 and result["orders_enabled"] is False
    con.close()


def test_immediate_diagnostic_is_separate_from_primary_bar_close_scores(tmp_path, monkeypatch):
    cfg = load_challengers()
    con = database(tmp_path)
    q = {
        "symbol": "BTCUSDT",
        "observed_utc": "2026-11-16T13:30:00Z",
        "bid": 100.0,
        "ask": 100.01,
        "mid": 100.005,
        "spread_bps": 1.0,
    }
    import sentinel_crypto.challenger_watch as watch

    monkeypatch.setattr(watch, "probabilities", lambda *a: {"lstm": 0.6})
    history = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2026-11-16T08:00:00Z")])
    result = observe(
        con, {"BTCUSDT": {"boundary": "2026-11-16T12:00:00Z"}}, {"BTCUSDT": history}, cfg, tmp_path, quotes=[q]
    )
    assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM startup_diagnostics").fetchone()[0] == 1
    assert result["books"] == []
    assert result["startup_diagnostic_books"][0]["pending"] == 1
    con.close()


@pytest.mark.parametrize("name", ["lstm", "tft"])
def test_neural_fit_predict_and_serialization_keep_actual_architecture(name, tmp_path):
    pytest.importorskip("pytorch_forecasting")
    f, y, ends, boundary = fixture_data()
    cfg = {**load_challengers(), "max_epochs": 1, "max_training_sequences": 128, "batch_size": 64}
    model = fit_challenger(name, f, y, ends, boundary, cfg)
    p = model.predict(f, [1200, 1206])
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    if name == "tft":
        assert type(model._net).__name__ == "TemporalFusionTransformer"
    else:
        assert model._net.recurrent.num_layers == 2
    path = tmp_path / f"{name}.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(f, [1200, 1206]), p, atol=1e-7)
