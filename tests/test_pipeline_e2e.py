"""End-to-end: dataset -> walk-forward -> model bundle -> backtest and live-path replay."""
import numpy as np
import pandas as pd
import pytest

from sentinel.backtest.engine import run_simulation
from sentinel.events import Bar, SignalSnapshot
from sentinel.features.technical import compute_all
from sentinel.market_time import NY
from sentinel.models.dataset import build_dataset
from sentinel.models.gbm import train_final, walk_forward
from sentinel.models.sentiment import LexiconSentiment, SentimentState
from sentinel.strategy.signals import WINDOW, LiveSignals, PrecomputedSignals


@pytest.fixture(scope="module")
def built(synth):
    from sentinel.config import Config
    cfg = Config(symbols=["AAA", "BBB", "CCC"], timeframe_minutes=15)
    cfg.model.min_train_bars, cfg.model.test_bars = 600, 300
    cfg.model.lgbm = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 30,
                      "verbose": -1}
    bars, news = synth
    lex = LexiconSentiment()
    scores = {n.id: lex.score(n.headline) for n in news}
    df = build_dataset(bars, news, scores, cfg)
    oos, wf = walk_forward(df, cfg)
    return cfg, bars, news, scores, df, oos, wf


def test_walk_forward_is_purged(built):
    cfg, _, _, _, df, oos, wf = built
    ts = np.sort(df["ts"].unique())
    gap = cfg.labels.horizon_bars + cfg.model.embargo_bars
    assert oos["ts"].min() == pd.Timestamp(ts[cfg.model.min_train_bars])
    for f in range(len(wf)):
        start = cfg.model.min_train_bars + f * cfg.model.test_bars
        train_rows = df[(df["ts"] < ts[start - gap]) & df["entry_window"] & df["label"].notna()]
        assert wf.loc[f, "train_rows"] == len(train_rows)
        assert oos[oos["fold"] == f]["ts"].min() == pd.Timestamp(ts[start])
    assert 0.0 <= oos["prob"].min() and oos["prob"].max() <= 1.0


def test_sentiment_features_only_use_published_news(built):
    cfg, _, news, scores, df, _, _ = built
    n = news[10]
    sym = n.symbols[0]
    rows = df[df["symbol"] == sym]
    before = rows[rows["ts"] + pd.Timedelta(minutes=15) < n.ts].iloc[-1]
    after = rows[rows["ts"] + pd.Timedelta(minutes=15) >= n.ts].iloc[0]
    if scores[n.id] != 0:
        assert before["sent_score"] != pytest.approx(after["sent_score"])


async def test_backtest_runs_and_never_holds_overnight(built):
    cfg, bars, news, scores, _, oos, _ = built
    feats = compute_all(bars)
    sim_bars = bars[(bars["ts"] >= oos["ts"].min()) & (bars["ts"] <= oos["ts"].max())]
    res = await run_simulation(cfg, sim_bars, news, PrecomputedSignals(oos, feats, cfg), scores=scores)
    assert not res.equity.empty and "sharpe" in res.metrics
    eq = res.equity.copy()
    eq["date"] = eq["ts"].dt.tz_convert(NY).dt.date
    last_of_day = eq.groupby("date").tail(1)
    assert (last_of_day["n_positions"] == 0).all()
    assert (res.equity["equity"] > 0).all()


class AlwaysLong:
    """Signal provider that always wants in: exercises the risk layer."""
    def __init__(self, atr=0.5):
        self.atr = atr

    def on_bar(self, b: Bar) -> SignalSnapshot:
        return SignalSnapshot(b.symbol, b.end, ready=True, prob=0.7, atr=self.atr, close=b.close, regime=1)


async def test_risk_limits_hold_under_aggressive_signals(built):
    cfg, bars, news, scores, *_ = built
    cfg = cfg.model_copy(deep=True)
    cfg.risk.max_open_positions = 2
    res = await run_simulation(cfg, bars.iloc[: 3 * 26 * 10], [], AlwaysLong())
    assert res.equity["n_positions"].max() <= 2
    entries = res.fills[res.fills["side"] == "buy"]
    assert len(entries) > 0
    ny_t = entries["ts"].dt.tz_convert(NY)
    mins = ny_t.dt.hour * 60 + ny_t.dt.minute - 570
    assert mins.min() >= cfg.risk.no_entry_minutes_after_open
    assert (390 - mins).min() >= cfg.risk.no_entry_minutes_before_close


async def test_drawdown_halt_in_simulation(cfg):
    """A crash right after entry must trigger the daily halt and block further entries that day."""
    from tests.conftest import make_bars, ny
    cfg.risk.max_position_pct = 1.0
    cfg.risk.risk_per_trade = 0.02
    cfg.risk.daily_drawdown_halt = 0.01
    rows = [(100, 100.2, 99.9, 100)] * 4 + [(100, 100, 60, 61)] + [(61, 61.5, 60.5, 61)] * 8
    b = make_bars("AAA", ny(2024, 3, 4, 9, 30), rows)
    res = await run_simulation(cfg, b, [], AlwaysLong(atr=20))
    assert any(res.risk_events["rule"] == "daily_drawdown")
    assert (res.fills["side"] == "buy").sum() == 1


def test_train_serve_parity(built):
    """The live path (rolling window + per-bar inference) must reproduce batch predictions exactly."""
    cfg, bars, news, scores, df, _, wf = built
    bundle = train_final(df, cfg, wf)
    feats = df[df["symbol"] == "AAA"].tail(5)
    batch_prob, _ = bundle.predict(feats)

    state = SentimentState(cfg.sentiment.decay_minutes)
    live = LiveSignals(bundle, state, cfg)
    one = bars[bars["symbol"] == "AAA"].sort_values("ts")
    first_live = one[one["ts"] >= feats["ts"].iloc[0]]
    live.warm_up(one[one["ts"] < feats["ts"].iloc[0]].tail(WINDOW))
    end_of = lambda ts: ts + pd.Timedelta(minutes=15)
    pending = sorted((n for n in news if n.id in scores), key=lambda n: n.ts)
    j = 0   # the first iteration replays all earlier headlines, as the batch pipeline did
    live_probs = []
    for r in first_live.itertuples(index=False):
        end = end_of(r.ts).to_pydatetime()
        while j < len(pending) and pending[j].ts <= end:
            state.add(pending[j].symbols, pending[j].ts, scores[pending[j].id])
            j += 1
        s = live.on_bar(Bar(r.symbol, r.ts.to_pydatetime(), r.open, r.high, r.low, r.close, r.volume, 15))
        live_probs.append(s.prob)
    np.testing.assert_allclose(live_probs, batch_prob, rtol=1e-6)
