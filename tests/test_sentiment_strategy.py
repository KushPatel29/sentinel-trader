import math
from datetime import timedelta

from sentinel.events import Action, SignalSnapshot
from sentinel.models.sentiment import LexiconSentiment, SentimentState
from sentinel.strategy.ensemble import decide, expected_r
from tests.conftest import ny


def test_lexicon_direction_and_negation():
    s = LexiconSentiment()
    assert s.score("Acme beats estimates and raises guidance") > 0.5
    assert s.score("Acme misses estimates, faces SEC investigation") < -0.5
    assert s.score("Acme schedules earnings call") == 0.0
    assert s.score("Acme did not miss estimates") > 0


def test_sentiment_state_decays_and_averages():
    st = SentimentState(decay_minutes=60)
    t0 = ny(2024, 3, 4, 10, 0)
    st.add(("X",), t0, -0.9)
    assert math.isclose(st.score("X", t0), -0.9)
    assert math.isclose(st.score("X", t0 + timedelta(minutes=60)), -0.9 * math.exp(-1), rel_tol=1e-9)
    st.add(("X",), t0, +0.9)                  # two opposite headlines at once average out
    assert abs(st.score("X", t0)) < 1e-9
    assert st.score("Y", t0) == 0.0


def _sig(prob, regime_blocked=False, ready=True):
    return SignalSnapshot("X", ny(2024, 3, 4, 11, 0), ready=ready, prob=prob, atr=1.0, close=100.0,
                          regime=1, regime_blocked=regime_blocked)


def test_decision_rules(cfg):
    assert decide(_sig(0.6), 0.0, False, cfg).action == Action.ENTER_LONG
    assert decide(_sig(0.3), 0.0, False, cfg).action == Action.HOLD                  # below threshold
    assert decide(_sig(0.6, regime_blocked=True), 0.0, False, cfg).action == Action.HOLD
    assert decide(_sig(0.6), -0.8, False, cfg).action == Action.HOLD                 # news veto
    assert decide(_sig(0.6), -0.8, True, cfg).action == Action.EXIT                  # veto exits a holding
    assert decide(_sig(0.6), 0.8, False, cfg).size_mult == cfg.sentiment.boost_multiplier
    assert decide(_sig(float("nan"), ready=False), 0.0, False, cfg).action == Action.HOLD


def test_expected_r():
    assert math.isclose(expected_r(1 / 3, 2.0), 0.0, abs_tol=1e-12)    # break-even at 2:1
    assert expected_r(0.5, 2.0) == 0.5
