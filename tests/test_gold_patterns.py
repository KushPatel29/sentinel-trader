"""Pattern engine: causal (no lookahead), detects textbook shapes, analog matching uses only known outcomes."""
import numpy as np
import pandas as pd

from sentinel.gold.patterns import PATTERN_INFO, all_patterns, analog_forecast, zigzag


def _bars(n=900, seed=5):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = c * np.exp(rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.005, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1e5},
                        index=pd.bdate_range("2015-01-01", periods=n))


def test_patterns_have_no_lookahead():
    df = _bars()
    base = all_patterns(df)
    cut = 600
    alt = df.copy()
    alt.iloc[cut + 1:, :4] *= 1.5
    pd.testing.assert_frame_equal(base.iloc[:cut + 1], all_patterns(alt).iloc[:cut + 1])
    assert set(base.columns) == set(PATTERN_INFO)


def test_zigzag_pivots_alternate_and_are_confirmed_later():
    pv = zigzag(_bars())
    assert len(pv) > 10
    assert all(a.kind != b.kind for a, b in zip(pv, pv[1:]))
    assert all(p.confirm > p.i for p in pv)


def test_double_bottom_detected():
    # flat, drop to 90, bounce to 100, drop to ~90, then break above 100
    path = [100] * 40 + list(np.linspace(100, 90, 10)) + list(np.linspace(90, 100, 10)) + \
        list(np.linspace(100, 90.3, 10)) + list(np.linspace(90.3, 104, 12)) + [104] * 5
    c = pd.Series(path, index=pd.bdate_range("2020-01-01", periods=len(path)), dtype=float)
    df = pd.DataFrame({"open": c.shift(1).fillna(c), "high": c + 0.6, "low": c - 0.6, "close": c, "volume": 1.0})
    s = all_patterns(df)
    assert s["double_bottom"].sum() >= 1
    hit = s.index[s["double_bottom"] > 0][0]
    assert df.loc[hit, "close"] > 100            # confirmed only on the neckline break


def test_analog_forecast_shapes():
    df = _bars(1400)
    a = analog_forecast(df["close"], horizon=5, min_history=1000)
    assert len(a) > 300 and a["analog_up_share"].between(0, 1).all()
    last = analog_forecast(df["close"], horizon=5, min_history=1000, start=df.index[-1])
    assert len(last) == 1 and last.index[0] == df.index[-1]
