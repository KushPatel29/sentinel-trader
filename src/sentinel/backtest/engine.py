"""Event-driven backtest / replay driver.

Feeds bars and news in strict time order through the real TradingEngine with a SimBroker.
For each bar timestamp T (bars cover [T, T+N)):
  1. news published before T+N is delivered
  2. the simulator works open brackets against each bar's open/high/low and marks to the close
  3. the engine's timer runs (drawdown halt, end-of-day flatten, equity snapshot)
  4. the engine decides on each completed bar; entries fill at that bar's close + slippage"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pandas as pd

from sentinel.backtest.metrics import buy_and_hold, performance
from sentinel.config import Config
from sentinel.data.store import Store
from sentinel.engine import TradingEngine
from sentinel.events import Bar, NewsItem
from sentinel.execution.broker import SimBroker
from sentinel.models.sentiment import SentimentAnalyzer, SentimentState
from sentinel.risk.officer import RiskOfficer

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    fills: pd.DataFrame
    decisions: pd.DataFrame
    risk_events: pd.DataFrame
    metrics: dict
    benchmark: dict
    extra: dict = field(default_factory=dict)


async def run_simulation(cfg: Config, bars: pd.DataFrame, news: list[NewsItem], signals,
                         scores: dict[str, float] | None = None, analyzer: SentimentAnalyzer | None = None,
                         store: Store | None = None, progress_every: int = 0,
                         sentiment_state: SentimentState | None = None) -> BacktestResult:
    """`sentiment_state`: pass the same state a LiveSignals provider reads, so model features and the
    news veto see identical sentiment (as they do in live trading)."""
    cap = cfg.backtest.initial_capital
    broker = SimBroker(cfg, cap)
    risk = RiskOfficer(cfg, on_event=(lambda e: store.log_risk(e.ts, e.level, e.rule, e.message)) if store else None)
    sentiment = sentiment_state or SentimentState(cfg.sentiment.decay_minutes)
    engine = TradingEngine(cfg, broker, signals, sentiment, risk, analyzer=analyzer, store=store,
                           timer_step_minutes=cfg.timeframe_minutes)
    minutes = cfg.timeframe_minutes
    news = sorted(news, key=lambda n: n.ts)
    j = 0
    groups = bars.sort_values(["ts", "symbol"]).groupby("ts", sort=True)
    for k, (ts, g) in enumerate(groups):
        end = (ts + timedelta(minutes=minutes)).to_pydatetime()
        while j < len(news) and news[j].ts <= end:
            n = news[j]
            broker.now = n.ts
            await engine.on_news(n, n.ts, score=scores.get(n.id) if scores is not None else None)
            j += 1
        broker.now = end
        bar_objs = [Bar(r.symbol, r.ts.to_pydatetime(), r.open, r.high, r.low, r.close, r.volume, minutes)
                    for r in g.itertuples(index=False)]
        for b in bar_objs:
            for f in broker.process_bar(b):
                if store:
                    store.log_fill(f.ts, f.symbol, f.side, f.qty, f.price, f.reason, f.pnl)
        await engine.on_timer(end)
        for b in bar_objs:
            await engine.on_bar(b, end)
        if progress_every and k % progress_every == 0:
            log.info("%s equity=%.2f positions=%d", end, broker.equity(), len(broker.positions))
        if engine.stopped:
            break

    eq = pd.DataFrame([e.__dict__ for e in engine.equity_curve])
    fills = pd.DataFrame([f.__dict__ for f in broker.fills],
                         columns=["ts", "symbol", "side", "qty", "price", "reason", "pnl"])
    dec = pd.DataFrame(engine.decisions, columns=["ts", "symbol", "action", "reason"])
    rev = pd.DataFrame([e.__dict__ for e in risk.state.events])
    metrics = performance(eq, fills, cap)
    metrics["fees_paid"] = broker.fees_paid
    bench = buy_and_hold(bars, eq["ts"].min(), eq["ts"].max(), cap) if not eq.empty else {}
    return BacktestResult(eq, fills, dec, rev, metrics, bench)


def save_report(res: BacktestResult, out_dir: Path, title: str, extra: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(out_dir / "equity.csv", index=False)
    res.fills.to_csv(out_dir / "fills.csv", index=False)
    res.risk_events.to_csv(out_dir / "risk_events.csv", index=False)
    entries = res.decisions[res.decisions["action"] != "hold"]
    entries.to_csv(out_dir / "decisions.csv", index=False)
    payload = {"title": title, "strategy": res.metrics, "buy_and_hold": res.benchmark, **(extra or {})}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    if not res.equity.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[3, 1])
        eq = res.equity.set_index("ts")["equity"]
        ax1.plot(eq.index, eq.values, lw=1.2, color="#2a6fdb", label="strategy")
        ax1.set_title(title)
        ax1.set_ylabel("equity ($)")
        ax1.grid(alpha=0.3)
        dd = eq / eq.cummax() - 1
        ax2.fill_between(dd.index, dd.values * 100, 0, color="#d64545", alpha=0.5)
        ax2.set_ylabel("drawdown %")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "equity.png", dpi=130)
        plt.close(fig)
    return out_dir
