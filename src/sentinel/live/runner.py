"""Live / paper trading on Alpaca.

Three websocket streams (minute bars, news, order updates) run in their own threads and hand events
to one asyncio queue. A consumer task feeds the engine; a timer task runs the risk checks, flushes
bars, and watches the data feed's heartbeat. Blocking broker calls run in worker threads."""
from __future__ import annotations

import asyncio
import logging
import signal
import threading
from datetime import timedelta

from sentinel.config import Config, Secrets
from sentinel.data.alpaca_data import AlpacaData, news_from_alpaca
from sentinel.data.store import Store
from sentinel.engine import TradingEngine
from sentinel.events import Bar
from sentinel.execution.broker import AlpacaBroker
from sentinel.live.aggregator import BarAggregator
from sentinel.market_time import in_regular_session, utcnow
from sentinel.models.gbm import ModelBundle
from sentinel.models.sentiment import SentimentAnalyzer, SentimentState
from sentinel.risk.officer import RiskOfficer
from sentinel.strategy.signals import WINDOW, LiveSignals

log = logging.getLogger(__name__)

TIMER_S = 5
FEED_SILENCE_S = 180


def _start_stream(stream, name: str) -> threading.Thread:
    def target():
        try:
            stream.run()          # alpaca-py reconnects internally
        except Exception:
            log.exception("%s stream crashed", name)
    t = threading.Thread(target=target, name=f"stream-{name}", daemon=True)
    t.start()
    return t


def check_live_allowed(cfg: Config, secrets: Secrets, confirm_live: bool) -> bool:
    """Returns paper flag. Real money needs three independent switches."""
    if cfg.mode == "paper":
        return True
    if not (confirm_live and secrets.live_confirm):
        raise SystemExit("mode: live requires BOTH the --confirm-live flag and SENTINEL_LIVE_CONFIRM=yes. "
                         "Refusing to trade real money.")
    log.warning("LIVE TRADING WITH REAL MONEY ENABLED")
    return False


async def run_live(cfg: Config, secrets: Secrets, confirm_live: bool = False) -> None:
    paper = check_live_allowed(cfg, secrets, confirm_live)
    key, secret = secrets.require_alpaca()
    store = Store(cfg.data.db_path)
    bundle = ModelBundle.load(cfg.model.path)
    if bundle.meta.get("timeframe_minutes") != cfg.timeframe_minutes:
        raise SystemExit("model was trained on a different timeframe; run `sentinel train` again")
    log.info("model trained %s on %s", bundle.meta.get("trained_at"), bundle.meta.get("symbols"))

    data = AlpacaData(key, secret, cfg.data.feed)
    broker = AlpacaBroker(key, secret, paper=paper)
    sentiment = SentimentState(cfg.sentiment.decay_minutes)
    analyzer = SentimentAnalyzer(cfg.sentiment, secrets.anthropic_key)
    risk = RiskOfficer(cfg, on_event=lambda e: store.log_risk(e.ts, e.level, e.rule, e.message))
    signals = LiveSignals(bundle, sentiment, cfg)
    engine = TradingEngine(cfg, broker, signals, sentiment, risk, analyzer=analyzer, store=store,
                           react_to_news_immediately=True, clock_ttl_s=30)

    # ---- warm-up: price history for rolling features, recent headlines for the sentiment state
    log.info("warming up with %d bars per symbol", WINDOW)
    hist = await asyncio.to_thread(data.recent_bars, cfg.symbols, cfg.timeframe_minutes, WINDOW)
    now = utcnow()
    # drop a still-forming bar so it is not treated as complete
    hist = hist[hist["ts"] + timedelta(minutes=cfg.timeframe_minutes) <= now]
    signals.warm_up(hist)
    store.upsert_bars_df(hist, cfg.timeframe_minutes)
    lookback = now - timedelta(minutes=cfg.sentiment.decay_minutes * 4)
    for item in await asyncio.to_thread(data.fetch_news, cfg.symbols, lookback, now):
        await engine.on_news(item, now)
    await engine.refresh()
    acct = engine.account
    log.info("account equity %.2f, buying power %.2f, open positions %s (%s)", acct.equity, acct.buying_power,
             list(engine.positions), "PAPER" if paper else "LIVE")
    if engine.positions:
        log.warning("existing positions in managed symbols will be flattened at the close: %s",
                    list(engine.positions))

    # ---- streams
    from alpaca.data.enums import DataFeed
    from alpaca.data.live import StockDataStream
    from alpaca.data.live.news import NewsDataStream
    from alpaca.trading.stream import TradingStream

    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def push(kind):
        async def handler(obj):
            loop.call_soon_threadsafe(q.put_nowait, (kind, obj))
        return handler

    bars_stream = StockDataStream(key, secret, feed=DataFeed.SIP if cfg.data.feed == "sip" else DataFeed.IEX)
    bars_stream.subscribe_bars(push("bar"), *cfg.symbols)
    news_stream = NewsDataStream(key, secret)
    news_stream.subscribe_news(push("news"), *cfg.symbols)
    trade_stream = TradingStream(key, secret, paper=paper)
    trade_stream.subscribe_trade_updates(push("trade"))
    streams = [(bars_stream, "bars"), (news_stream, "news"), (trade_stream, "trades")]
    for s, name in streams:
        _start_stream(s, name)

    agg = BarAggregator(cfg.timeframe_minutes)
    stop = asyncio.Event()
    last_minute_bar = {"t": utcnow()}

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:   # Windows
            pass

    async def handle_bar(b: Bar):
        store.upsert_bar(b)
        d = await engine.on_bar(b, utcnow())
        if d.action.value != "hold":
            log.info("%s %s: %s", b.symbol, d.action.value, d.reason)

    async def consumer():
        while not stop.is_set():
            kind, obj = await q.get()
            try:
                if kind == "bar":
                    last_minute_bar["t"] = utcnow()
                    m = Bar(obj.symbol, obj.timestamp, float(obj.open), float(obj.high), float(obj.low),
                            float(obj.close), float(obj.volume), 1)
                    for done in agg.add(m):
                        await handle_bar(done)
                elif kind == "news":
                    item = news_from_alpaca(obj)
                    s = await engine.on_news(item, utcnow())
                    log.info("news %+.2f %s: %s", s, ",".join(item.symbols), item.headline[:100])
                elif kind == "trade":
                    o = obj.order
                    if obj.event in ("fill", "partial_fill"):
                        store.log_fill(obj.timestamp, o.symbol, str(o.side.value), float(obj.qty or 0),
                                       float(obj.price or 0), f"{o.order_type.value} {obj.event}")
                    log.info("order %s %s %s", o.symbol, obj.event, o.status)
            except Exception:
                log.exception("error handling %s event", kind)

    async def timer():
        warned = False
        while not stop.is_set():
            now = utcnow()
            try:
                for done in agg.flush_due(now):
                    await handle_bar(done)
                await engine.on_timer(now)
                if engine.stopped:
                    stop.set()
                silent = (now - last_minute_bar["t"]).total_seconds()
                clock = engine.clock
                if clock and clock.is_open and in_regular_session(now) and silent > FEED_SILENCE_S:
                    if not warned:
                        risk._emit(now, "warning", "feed_silence", f"no minute bars for {silent:.0f}s")
                        warned = True
                else:
                    warned = False
            except Exception:
                log.exception("timer error")
            await asyncio.sleep(TIMER_S)

    log.info("Sentinel running (%s). Ctrl+C to stop; create '%s' to flatten and halt.",
             "paper" if paper else "LIVE", cfg.risk.kill_switch_file)
    tasks = [asyncio.create_task(consumer()), asyncio.create_task(timer())]
    await stop.wait()
    log.info("shutting down")
    for t in tasks:
        t.cancel()
    for s, _ in streams:
        try:
            s.stop()
        except Exception:
            pass
    await engine.shutdown(flatten=cfg.execution.flatten_on_shutdown)
    store.close()
