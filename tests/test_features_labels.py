import numpy as np
import pandas as pd

from sentinel.features.labels import triple_barrier
from sentinel.features.technical import TECH_FEATURES, compute_features
from tests.conftest import make_bars, ny


def test_features_have_no_lookahead(synth):
    bars, _ = synth
    one = bars[bars["symbol"] == "AAA"].reset_index(drop=True)
    base = compute_features(one)
    cut = 500
    tampered = one.copy()
    tampered.loc[cut + 1:, ["open", "high", "low", "close"]] *= 3.0      # change only the future
    tampered.loc[cut + 1:, "volume"] *= 10
    alt = compute_features(tampered)
    cols = TECH_FEATURES + ["atr"]
    pd.testing.assert_frame_equal(base.loc[:cut, cols], alt.loc[:cut, cols])


def test_rolling_window_matches_full_history(synth):
    """Live inference computes features on a 300-bar window; they must match the training features."""
    bars, _ = synth
    one = bars[bars["symbol"] == "BBB"].reset_index(drop=True)
    full = compute_features(one).iloc[-1]
    window = compute_features(one.tail(300).reset_index(drop=True)).iloc[-1]
    for c in TECH_FEATURES:
        assert np.isclose(full[c], window[c], rtol=1e-4, atol=1e-7, equal_nan=True), c


def _labels(rows, horizon=3, tp=2.0, sl=1.0, atr=1.0):
    df = make_bars("X", ny(2024, 3, 4, 10, 0), rows)
    return triple_barrier(df, pd.Series(atr, index=df.index), horizon, tp, sl).tolist()


def test_take_profit_first_is_positive():
    # entry close 100 -> tp 102, sl 99. Next bar reaches 102.5 without touching 99.
    lab = _labels([(100, 100, 100, 100), (100, 102.5, 99.5, 102), (102, 102, 102, 102), (102, 102, 102, 102)])
    assert lab[0] == 1.0


def test_stop_first_is_negative_and_same_bar_counts_as_stop():
    lab = _labels([(100, 100, 100, 100), (100, 100.5, 98.9, 99), (99, 103, 99, 103), (103, 103, 103, 103)])
    assert lab[0] == 0.0
    both = _labels([(100, 100, 100, 100), (100, 102.5, 98.5, 101), (101, 101, 101, 101), (101, 101, 101, 101)])
    assert both[0] == 0.0


def test_timeout_is_negative_and_last_bar_undefined():
    lab = _labels([(100, 100, 100, 100)] * 5)
    assert lab[0] == 0.0
    assert np.isnan(lab[-1])


def test_labels_never_cross_the_session_close():
    df = pd.concat([
        make_bars("X", ny(2024, 3, 4, 15, 45), [(100, 100, 100, 100)]),
        make_bars("X", ny(2024, 3, 5, 9, 30), [(100, 105, 100, 105)]),     # next morning rally
    ], ignore_index=True)
    lab = triple_barrier(df, pd.Series(1.0, index=df.index), 4, 2.0, 1.0)
    assert np.isnan(lab.iloc[0])     # last bar of the day has no same-session future
