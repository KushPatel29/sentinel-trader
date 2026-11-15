"""Signal fusion: the one place that turns model outputs into a trade decision.

Pure function, shared verbatim by the backtester, replay mode and live trading.
  * ML probability + expected-value check decide whether a long setup exists
  * the regime filter switches entries off in the most volatile state
  * news sentiment is a veto (and an exit trigger) when strongly negative, and a size boost when
    strongly positive"""
from __future__ import annotations

import math

from sentinel.config import Config
from sentinel.events import Action, Decision, SignalSnapshot


def expected_r(prob: float, payoff: float) -> float:
    """Expected profit in units of risk (R) for a bracket with reward:risk = payoff."""
    return prob * payoff - (1 - prob)


def decide(sig: SignalSnapshot, sentiment: float, has_position: bool, cfg: Config) -> Decision:
    sc = cfg.sentiment
    if has_position:
        if sentiment <= sc.veto_threshold:
            return Decision(Action.EXIT, f"news veto exit (sentiment {sentiment:+.2f})")
        return Decision(Action.HOLD, "in position; bracket manages exit")

    if not sig.ready or math.isnan(sig.prob):
        return Decision(Action.HOLD, "features warming up")
    if sig.regime_blocked:
        return Decision(Action.HOLD, f"regime {sig.regime} blocked (high volatility)")
    if sentiment <= sc.veto_threshold:
        return Decision(Action.HOLD, f"news veto (sentiment {sentiment:+.2f})")
    if sig.prob < cfg.model.entry_threshold:
        return Decision(Action.HOLD, f"prob {sig.prob:.3f} < {cfg.model.entry_threshold}")
    edge = expected_r(sig.prob, cfg.labels.payoff_ratio)
    if edge < cfg.model.min_edge:
        return Decision(Action.HOLD, f"edge {edge:.3f}R < {cfg.model.min_edge}R")

    mult = sc.boost_multiplier if sentiment >= sc.boost_threshold else 1.0
    why = f"prob {sig.prob:.3f}, edge {edge:+.2f}R, regime {sig.regime}, sentiment {sentiment:+.2f}"
    return Decision(Action.ENTER_LONG, why + (" (news boost)" if mult > 1 else ""), size_mult=mult)
