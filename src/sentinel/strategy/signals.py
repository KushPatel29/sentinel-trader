"""Signal providers: turn a completed bar into a SignalSnapshot.

PrecomputedSignals  - backtests: looks up walk-forward out-of-sample probabilities.
LiveSignals         - replay & live: computes features on a rolling window and runs the model,
                      exactly as it would on a production trading day."""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

from sentinel.config import Config
from sentinel.events import Bar, SignalSnapshot
from sentinel.features.technical import WARMUP_BARS, compute_features
from sentinel.models.gbm import ModelBundle
from sentinel.models.sentiment import SentimentState

WINDOW = 300  # bars kept per symbol; comfortably above the longest look-back + EMA convergence


class PrecomputedSignals:
    def __init__(self, oos: pd.DataFrame, feats: pd.DataFrame, cfg: Config):
        merged = oos.merge(feats[["ts", "symbol", "atr", "close"]], on=["ts", "symbol"], how="left")
        self._map = {(r.symbol, r.ts.to_pydatetime()): r for r in merged.itertuples(index=False)}
        self.cfg = cfg

    def on_bar(self, bar: Bar) -> SignalSnapshot:
        r = self._map.get((bar.symbol, bar.ts))
        if r is None:
            return SignalSnapshot(bar.symbol, bar.end, ready=False, close=bar.close)
        blocked = self.cfg.regime.block_high_vol and r.regime == self.cfg.regime.n_states - 1
        return SignalSnapshot(bar.symbol, bar.end, ready=True, prob=float(r.prob), atr=float(r.atr),
                              close=bar.close, regime=int(r.regime), regime_blocked=blocked)


class LiveSignals:
    def __init__(self, bundle: ModelBundle, sentiment: SentimentState, cfg: Config):
        self.bundle = bundle
        self.sentiment = sentiment
        self.cfg = cfg
        self.windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.last_features: dict[str, dict] = {}

    def warm_up(self, bars: pd.DataFrame) -> None:
        for r in bars.sort_values("ts").itertuples(index=False):
            self.windows[r.symbol].append((r.ts.to_pydatetime(), r.open, r.high, r.low, r.close, r.volume))

    def on_bar(self, bar: Bar) -> SignalSnapshot:
        w = self.windows[bar.symbol]
        if w and w[-1][0] >= bar.ts:          # duplicate / out-of-order bar (e.g. after reconnect)
            if w[-1][0] == bar.ts:
                w.pop()
            else:
                return SignalSnapshot(bar.symbol, bar.end, ready=False, close=bar.close)
        w.append((bar.ts, bar.open, bar.high, bar.low, bar.close, bar.volume))
        if len(w) < WARMUP_BARS:
            return SignalSnapshot(bar.symbol, bar.end, ready=False, close=bar.close)
        df = pd.DataFrame(list(w), columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df.insert(0, "symbol", bar.symbol)
        feats = compute_features(df).iloc[[-1]].copy()
        feats["sent_score"] = self.sentiment.score(bar.symbol, bar.end)
        feats["sent_intensity"] = self.sentiment.intensity(bar.symbol, bar.end)
        prob, reg = self.bundle.predict(feats)
        row = feats.iloc[0]
        self.last_features[bar.symbol] = row.to_dict()
        regime = int(reg[0])
        blocked = self.cfg.regime.block_high_vol and regime == self.cfg.regime.n_states - 1
        return SignalSnapshot(bar.symbol, bar.end, ready=bool(pd.notna(row["atr"])), prob=float(prob[0]),
                              atr=float(row["atr"]), close=bar.close, regime=regime, regime_blocked=blocked)
