"""Classic rule-based strategies with fixed textbook parameters (no tuning, so no overfitting).

A position decided at the close of day t earns the return of day t+1. Turnover pays a cost that
depends on the asset class."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COST_BPS = {"stock": 5, "etf": 3, "index": 3, "crypto": 20, "commodity": 5, "fx": 3}


@dataclass(frozen=True)
class StrategySpec:
    key: str
    name: str
    family: str          # "trend" | "reversion" | "benchmark" | "ml" | "meta"
    description: str


SPECS = [
    StrategySpec("buy_hold", "Buy & hold", "benchmark", "Always invested. The bar every strategy must beat."),
    StrategySpec("sma200", "200-day trend filter", "trend", "Long while price is above its 200-day average."),
    StrategySpec("golden_cross", "Golden cross 50/200", "trend", "Long while the 50-day average is above the 200-day."),
    StrategySpec("time_momentum", "12-1 month momentum", "trend",
                 "Long when the 12-month return (skipping the last month) is positive."),
    StrategySpec("macd", "MACD trend", "trend", "Long while MACD is above its signal line."),
    StrategySpec("donchian", "Donchian breakout 55/20", "trend",
                 "Turtle rules: buy a 55-day high, sell a 20-day low."),
    StrategySpec("rsi2", "RSI(2) pullback", "reversion",
                 "Buy short, sharp dips (RSI(2) < 10) inside an uptrend; sell on a close above the 5-day average."),
    StrategySpec("bollinger", "Bollinger reversion", "reversion",
                 "Buy a close below the lower band in an uptrend; sell at the 20-day average."),
    StrategySpec("vol_target", "Volatility-targeted trend", "trend",
                 "200-day trend filter sized to 15% annual volatility (smaller in turbulent markets)."),
]


def _stateful(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    pos, out = 0.0, np.zeros(len(enter))
    e, x = enter.to_numpy(), exit_.to_numpy()
    for i in range(len(e)):
        if pos == 0 and e[i]:
            pos = 1.0
        elif pos == 1 and x[i]:
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=enter.index)


def positions(px: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    c, h, lo = px["close"], px["high"], px["low"]
    sma5, sma20, sma50, sma200 = (c.rolling(n).mean() for n in (5, 20, 50, 200))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    p = pd.DataFrame(index=px.index)
    p["buy_hold"] = 1.0
    p["sma200"] = (c > sma200).astype(float)
    p["golden_cross"] = (sma50 > sma200).astype(float)
    p["time_momentum"] = (f["mom_12_1"] > 0).astype(float)
    p["macd"] = (macd > sig).astype(float)
    p["donchian"] = _stateful(c > h.shift(1).rolling(55).max(), c < lo.shift(1).rolling(20).min())
    uptrend = c > sma200
    p["rsi2"] = _stateful(uptrend & (f["rsi2"] < 10), c > sma5)
    p["bollinger"] = _stateful(uptrend & (f["bb_pos"] < -1), c > sma20)
    vol = f["vol21"].clip(lower=0.05)
    p["vol_target"] = p["sma200"] * (0.15 / vol).clip(upper=1.0)
    warm = sma200.isna()
    p.loc[warm, [k for k in p.columns if k != "buy_hold"]] = 0.0
    return p


def strategy_returns(close: pd.Series, pos: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    r = close.pct_change().fillna(0.0)
    held = pos.shift(1).fillna(0.0)                 # decided at t, earns t+1
    turnover = pos.diff().abs().fillna(pos.abs())
    return held.mul(r, axis=0) - turnover.shift(1).fillna(0.0) * cost_bps / 1e4


def adaptive(rets: pd.DataFrame, lookback: int = 252, rebalance: int = 21) -> pd.Series:
    """Meta-strategy: every month, follow whichever strategy had the best trailing Sharpe.
    Uses only past returns, so it is an honest out-of-sample test of 'pick the winner'."""
    sharpe = rets.rolling(lookback, min_periods=126).mean() / rets.rolling(lookback, min_periods=126).std()
    choice = pd.Series(index=rets.index, dtype=object)
    current = "buy_hold"
    for i, d in enumerate(rets.index):
        if i % rebalance == 0 and sharpe.loc[d].notna().any():
            current = sharpe.loc[d].idxmax()
        choice.iloc[i] = current
    held = choice.shift(1).fillna("buy_hold")
    out = pd.Series([rets.at[d, k] for d, k in zip(rets.index, held)], index=rets.index)
    out.attrs["choice"] = choice
    return out


def perf(r: pd.Series, periods: int = 252, exposure: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / periods
    sd = r.std()
    down = r[r < 0].std()
    dd = eq / eq.cummax() - 1
    sharpe = float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0
    return {
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else -1.0,
        "total_return": float(eq.iloc[-1] - 1),
        "sharpe": sharpe,
        "sortino": float(r.mean() / down * np.sqrt(periods)) if down and down > 0 else 0.0,
        "max_drawdown": float(dd.min()),
        "volatility": float(sd * np.sqrt(periods)),
        "t_stat": float(r.mean() / sd * np.sqrt(len(r))) if sd > 0 else 0.0,
        "exposure": float((exposure > 0).mean()) if exposure is not None else 1.0,
        "days": int(len(r)),
    }
