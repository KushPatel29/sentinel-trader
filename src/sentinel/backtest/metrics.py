"""Performance statistics from an equity curve and a fill log."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.market_time import NY


def daily_returns(equity: pd.DataFrame) -> pd.Series:
    eq = equity.set_index("ts")["equity"]
    daily = eq.groupby(eq.index.tz_convert(NY).date).last()
    return daily.pct_change().dropna()


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    return float((series / peak - 1).min()) if len(series) else 0.0


def performance(equity: pd.DataFrame, fills: pd.DataFrame, initial: float) -> dict:
    if equity.empty:
        return {}
    eq = equity["equity"]
    rets = daily_returns(equity)
    n_days = max(1, len(rets))
    total = float(eq.iloc[-1] / initial - 1)
    cagr = float((eq.iloc[-1] / initial) ** (252 / n_days) - 1) if eq.iloc[-1] > 0 else -1.0
    sd = rets.std()
    downside = rets[rets < 0].std()
    sharpe = float(np.sqrt(252) * rets.mean() / sd) if sd and sd > 0 else 0.0
    sortino = float(np.sqrt(252) * rets.mean() / downside) if downside and downside > 0 else 0.0
    mdd = max_drawdown(eq)

    exits = fills[fills["side"] == "sell"] if not fills.empty else fills
    pnl = exits["pnl"].astype(float) if not exits.empty else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf") if len(wins) else 0.0
    by_reason = exits["reason"].str.replace(r" \(gap\)", "", regex=True).value_counts().to_dict() if len(exits) else {}
    return {
        "total_return": total, "cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "trading_days": int(n_days), "trades": int(len(pnl)),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": pf, "expectancy_per_trade": float(pnl.mean()) if len(pnl) else 0.0,
        "exposure": float((equity["n_positions"] > 0).mean()),
        "best_day": float(rets.max()) if len(rets) else 0.0, "worst_day": float(rets.min()) if len(rets) else 0.0,
        "exit_reasons": by_reason,
    }


def buy_and_hold(bars: pd.DataFrame, start, end, initial: float) -> dict:
    """Equal-weight buy-and-hold of the same symbols over the same window (the honest benchmark)."""
    b = bars[(bars["ts"] >= start) & (bars["ts"] <= end)]
    if b.empty:
        return {}
    px = b.pivot_table(index="ts", columns="symbol", values="close").ffill().dropna()
    curve = (px / px.iloc[0]).mean(axis=1) * initial
    eq = pd.DataFrame({"ts": curve.index, "equity": curve.values})
    rets = daily_returns(eq)
    sd = rets.std()
    return {"total_return": float(curve.iloc[-1] / initial - 1),
            "sharpe": float(np.sqrt(252) * rets.mean() / sd) if sd and sd > 0 else 0.0,
            "max_drawdown": max_drawdown(curve)}
