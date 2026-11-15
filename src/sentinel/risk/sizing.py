"""Position sizing: volatility targeting with a fractional-Kelly cap.

risk budget  = equity * min(risk_per_trade, kelly_fraction * kelly(p, b))
shares       = risk budget / stop distance,  stop distance = sl_atr * ATR
so every trade loses about the same fraction of equity at its stop, whatever the stock's volatility
(volatile names get fewer shares). The notional cap and the news multiplier are applied last."""
from __future__ import annotations

import math

from sentinel.config import Config


def kelly(prob: float, payoff: float) -> float:
    """Kelly fraction of capital to risk on a bet that wins `payoff` R with probability `prob`."""
    return max(0.0, prob - (1 - prob) / payoff)


def position_size(equity: float, price: float, atr: float, prob: float, cfg: Config, size_mult: float = 1.0,
                  available_notional: float | None = None) -> int:
    if not all(math.isfinite(x) and x > 0 for x in (equity, price, atr)):
        return 0
    rc = cfg.risk
    stop_dist = cfg.labels.sl_atr * atr
    risk_frac = min(rc.risk_per_trade, rc.kelly_fraction * kelly(prob, cfg.labels.payoff_ratio))
    if risk_frac <= 0:
        return 0
    shares = equity * risk_frac / stop_dist * size_mult
    cap = equity * rc.max_position_pct
    if available_notional is not None:
        cap = min(cap, max(0.0, available_notional))
    shares = min(shares, cap / price)
    return int(math.floor(shares))   # whole shares: Alpaca bracket orders do not take fractional qty
