"""Drives the real live runner with fake Alpaca streams/clients: websocket handlers -> queue ->
1-minute aggregation -> engine -> broker, then a kill-switch shutdown. No network access needed."""
import asyncio
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest

from sentinel.config import Config, Secrets
from sentinel.data.store import Store
from sentinel.events import ClockState
from sentinel.execution.broker import SimBroker
from sentinel.market_time import session_bounds
from sentinel.models.dataset import build_dataset
from sentinel.models.gbm import train_final
from sentinel.models.sentiment import LexiconSentiment
from tests.conftest import bar as mkbar
from tests.conftest import ny


@pytest.fixture
def live_env(tmp_path, monkeypatch, synth):
    monkeypatch.chdir(tmp_path)
    bars, news = synth
    cfg = Config(symbols=["AAA", "BBB"], timeframe_minutes=15)
    cfg.data.db_path = str(tmp_path / "live.db")
    cfg.model.path = str(tmp_path / "bundle.pkl")
    cfg.model.lgbm = {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 30, "verbose": -1}
    cfg.model.entry_threshold = 0.0      # force entries so the order path is exercised
    cfg.model.min_edge = -10
    cfg.regime.block_high_vol = False
    cfg.risk.no_entry_minutes_after_open = 0
    lex = LexiconSentiment()
    two = bars[bars["symbol"].isin(cfg.symbols)]
    df = build_dataset(two, news, {n.id: lex.score(n.headline) for n in news}, cfg)
    train_final(df, cfg).save(cfg.model.path)

    day = ny(2024, 6, 3, 9, 30)          # a Monday after the synthetic history
    clock = {"now": day}
    hist = two.copy()

    class FakeData:
        def __init__(self, *a, **k): ...
        def recent_bars(self, symbols, minutes, n):
            return hist.groupby("symbol").tail(n).reset_index(drop=True)
        def fetch_news(self, *a, **k):
            return []

    class FakeBroker(SimBroker):
        def __init__(self, *a, **k):
            super().__init__(cfg, 100_000)
            for s in cfg.symbols:
                self.last[s] = float(hist[hist["symbol"] == s]["close"].iloc[-1])
        async def get_clock(self):
            o, c = session_bounds(clock["now"])
            return ClockState(clock["now"], True, o, c)

    broker_holder = {}

    def make_broker(*a, **k):
        broker_holder["b"] = FakeBroker()
        return broker_holder["b"]

    class FakeBarStream:
        def __init__(self, *a, **k):
            self.handler = None
        def subscribe_bars(self, handler, *symbols):
            self.handler, self.symbols = handler, symbols
        def run(self):
            async def go():
                for i in range(46):                         # 9:30 .. 10:15 -> three 15-minute bars
                    ts = day + timedelta(minutes=i)
                    for s in self.symbols:
                        px = broker_holder["b"].last[s] * (1 + 0.0005 * (i % 3 - 1))
                        b = mkbar(s, ts, px, px * 1.001, px * 0.999, px, minutes=1)
                        broker_holder["b"].process_bar(b)
                        broker_holder["b"].now = ts + timedelta(minutes=1)
                        clock["now"] = ts + timedelta(seconds=61)
                        await self.handler(SimpleNamespace(symbol=s, timestamp=ts, open=px, high=px * 1.001,
                                                           low=px * 0.999, close=px, volume=1000))
                    await asyncio.sleep(0.005)
                await asyncio.sleep(0.5)
                (tmp_path / cfg.risk.kill_switch_file).touch()   # ask the bot to flatten and stop
            asyncio.run(go())
        def stop(self): ...

    class Idle:
        def __init__(self, *a, **k): ...
        def subscribe_news(self, *a): ...
        def subscribe_trade_updates(self, *a): ...
        def run(self): time.sleep(0.1)
        def stop(self): ...

    import sentinel.live.runner as runner
    monkeypatch.setattr(runner, "AlpacaData", FakeData)
    monkeypatch.setattr(runner, "AlpacaBroker", make_broker)
    monkeypatch.setattr(runner, "utcnow", lambda: clock["now"])
    monkeypatch.setattr(runner, "TIMER_S", 0.05)
    monkeypatch.setattr("alpaca.data.live.StockDataStream", FakeBarStream)
    monkeypatch.setattr("alpaca.data.live.news.NewsDataStream", Idle)
    monkeypatch.setattr("alpaca.trading.stream.TradingStream", Idle)
    return cfg, broker_holder


async def test_live_runner_end_to_end(live_env):
    from sentinel.live.runner import run_live
    cfg, holder = live_env
    await asyncio.wait_for(run_live(cfg, Secrets(alpaca_key="k", alpaca_secret="s")), timeout=60)

    store = Store(cfg.data.db_path)
    signals = store.read_table("signals")
    # 15-minute bars at 9:30, 9:45 and 10:00 for both symbols reached the engine
    assert len(signals) >= 6
    orders = store.read_table("orders")
    assert len(orders) >= 1 and orders["client_order_id"].str.startswith("snt-").all()
    risk = store.read_table("risk_events")
    assert "kill_switch" in set(risk["rule"])
    assert not holder["b"].positions          # kill switch flattened everything
