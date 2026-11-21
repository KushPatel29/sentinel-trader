"""Command-line interface.

  sentinel fetch [--synthetic]       download bars + news into SQLite (or generate demo data)
  sentinel train [--holdout-days N]  walk-forward validation, then fit the production model
  sentinel backtest                  event-driven backtest on the walk-forward predictions
  sentinel replay [--days N]         run the live code path on stored history with a simulated broker
  sentinel live [--confirm-live]     trade on Alpaca (paper unless mode: live and confirmed)
  sentinel status                    account, positions and recent risk events
  sentinel flatten                   cancel orders and close positions in the configured symbols
  sentinel gold-plan [--dry-run]     execute the daily gold plan on Alpaca (GLD), once a day after the close
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sentinel.config import Config, Secrets, load_config
from sentinel.data.store import Store

log = logging.getLogger("sentinel")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("websockets", "urllib3", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _oos_path(cfg: Config) -> Path:
    return Path(cfg.model.path).with_name("oos_predictions.csv.gz")


# ---------------------------------------------------------------------------------------------- fetch
def cmd_fetch(cfg: Config, args) -> None:
    store = Store(cfg.data.db_path)
    if args.synthetic:
        from sentinel.data.synthetic import generate
        bars, news = generate(cfg.symbols, n_days=args.days or 300, minutes=cfg.timeframe_minutes)
        n = store.upsert_bars_df(bars, cfg.timeframe_minutes)
        store.upsert_news(news)
        log.info("generated %d synthetic bars and %d headlines -> %s", n, len(news), cfg.data.db_path)
        return
    from sentinel.data.alpaca_data import AlpacaData
    key, secret = Secrets.from_env().require_alpaca()
    data = AlpacaData(key, secret, cfg.data.feed)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)   # free IEX history lags real time slightly
    start = end - timedelta(days=args.days or cfg.data.history_days)
    bars = data.fetch_bars(cfg.symbols, start, end, cfg.timeframe_minutes)
    n = store.upsert_bars_df(bars, cfg.timeframe_minutes)
    news = data.fetch_news(cfg.symbols, start, end)
    store.upsert_news(news)
    log.info("stored %d bars and %d headlines for %s", n, len(news), cfg.symbols)


# ---------------------------------------------------------------------------------------------- train
def cmd_train(cfg: Config, args) -> None:
    from sentinel.models.gbm import train_final, walk_forward
    from sentinel.models.sentiment import SentimentAnalyzer
    from sentinel.pipeline import make_dataset

    store = Store(cfg.data.db_path)
    analyzer = SentimentAnalyzer(cfg.sentiment)
    df, bars, _, _ = make_dataset(cfg, store, analyzer)
    if args.holdout_days:
        cutoff = df["ts"].max() - pd.Timedelta(days=args.holdout_days)
        log.info("holding out data after %s (for `sentinel replay`)", cutoff)
        df = df[df["ts"] <= cutoff]
    log.info("dataset: %d rows, %d symbols, %s -> %s, base rate %.3f", len(df), df["symbol"].nunique(),
             df["ts"].min(), df["ts"].max(), df.loc[df["entry_window"], "label"].mean())

    oos, wf = walk_forward(df, cfg)
    _oos_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    oos.to_csv(_oos_path(cfg), index=False, compression="gzip")
    bundle = train_final(df, cfg, wf)
    bundle.meta["sentiment_engine"] = analyzer.engine
    bundle.save(cfg.model.path)

    out = Path(cfg.backtest.reports_dir) / f"train_{_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    wf.to_csv(out / "walk_forward.csv", index=False)
    (out / "model_meta.json").write_text(json.dumps(bundle.meta, indent=2, default=str))
    cols = [c for c in ["fold", "test_start", "train_rows", "test_rows", "base_rate", "auc",
                        "precision_at_threshold", "signals"] if c in wf.columns]
    print("\nWalk-forward (out-of-sample) results:")
    print(wf[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nmean AUC {wf['auc'].mean():.3f} | model -> {cfg.model.path} | report -> {out}")
    print("top features:", ", ".join(k for k, _ in bundle.meta["feature_importance"][:8]))


# ---------------------------------------------------------------------------------------------- backtest
def cmd_backtest(cfg: Config, args) -> None:
    from sentinel.backtest.engine import run_simulation, save_report
    from sentinel.features.technical import compute_all
    from sentinel.models.sentiment import SentimentAnalyzer
    from sentinel.pipeline import load_history, score_news
    from sentinel.strategy.signals import PrecomputedSignals

    p = _oos_path(cfg)
    if not p.exists():
        raise SystemExit("No walk-forward predictions found. Run `sentinel train` first.")
    store = Store(cfg.data.db_path)
    oos = pd.read_csv(p, parse_dates=["ts"])
    oos["ts"] = pd.to_datetime(oos["ts"], utc=True)
    bars, news = load_history(cfg, store)
    scores = score_news(store, news, SentimentAnalyzer(cfg.sentiment))
    # trade only where predictions are out-of-sample
    bars = bars[(bars["ts"] >= oos["ts"].min()) & (bars["ts"] <= oos["ts"].max())]
    _quiet_sim_logs(args)
    feats = compute_all(store.load_bars(cfg.symbols, cfg.timeframe_minutes))
    signals = PrecomputedSignals(oos, feats, cfg)
    res = asyncio.run(run_simulation(cfg, bars, news, signals, scores=scores))
    out = save_report(res, Path(cfg.backtest.reports_dir) / f"backtest_{_stamp()}",
                      "Sentinel walk-forward backtest (out-of-sample)",
                      extra={"slippage_bps": cfg.execution.slippage_bps, "symbols": cfg.symbols})
    _print_metrics(res.metrics, res.benchmark)
    print(f"\nreport -> {out}")


def _quiet_sim_logs(args) -> None:
    if not args.verbose:   # per-trade logs are noise in a simulation; they are in the report files
        for name in ("sentinel.engine", "sentinel.risk.officer"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _print_metrics(m: dict, bench: dict) -> None:
    pct = {"total_return", "cagr", "max_drawdown", "win_rate", "exposure", "best_day", "worst_day"}
    print("\nStrategy:")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:22s} {v:>10.2%}" if k in pct else f"  {k:22s} {v:>10.3f}")
        else:
            print(f"  {k:22s} {v}")
    if m.get("trading_days", 0) < 20:
        print("  (fewer than 20 trading days: annualised ratios like Sharpe/CAGR are not meaningful)")
    if bench:
        print("Equal-weight buy & hold, same period:")
        for k, v in bench.items():
            print(f"  {k:22s} {v:>10.2%}" if k != "sharpe" else f"  {k:22s} {v:>10.3f}")


# ---------------------------------------------------------------------------------------------- replay
def cmd_replay(cfg: Config, args) -> None:
    """Runs the *live* signal path (rolling features + model inference + sentiment scoring per headline)
    over the last N days of stored data, with a simulated broker. Use `train --holdout-days N` first so
    those days are out-of-sample."""
    from sentinel.backtest.engine import run_simulation, save_report
    from sentinel.models.gbm import ModelBundle
    from sentinel.models.sentiment import SentimentAnalyzer, SentimentState
    from sentinel.strategy.signals import WINDOW, LiveSignals

    store = Store(cfg.data.db_path)
    bundle = ModelBundle.load(cfg.model.path)
    bars = store.load_bars(cfg.symbols, cfg.timeframe_minutes)
    start = bars["ts"].max().normalize() - pd.Timedelta(days=args.days - 1)
    warm = bars[bars["ts"] < start].groupby("symbol").tail(WINDOW)
    run = bars[bars["ts"] >= start]
    news = store.load_news(start=start.to_pydatetime())
    trained_until = pd.Timestamp(bundle.meta["data_end"])
    if trained_until >= start:
        log.warning("model saw data up to %s, which overlaps the replay window -> results are in-sample. "
                    "Run `sentinel train --holdout-days %d` first.", trained_until.date(), args.days)
    replay_path = Path(cfg.data.db_path).with_name("replay.db")
    replay_path.unlink(missing_ok=True)
    replay_db = Store(replay_path)
    _quiet_sim_logs(args)
    analyzer = SentimentAnalyzer(cfg.sentiment)
    state = SentimentState(cfg.sentiment.decay_minutes)   # shared by the model features and the news veto
    signals = LiveSignals(bundle, state, cfg)
    signals.warm_up(warm)
    res = asyncio.run(run_simulation(cfg, run, news, signals, analyzer=analyzer, store=replay_db,
                                     sentiment_state=state))
    out = save_report(res, Path(cfg.backtest.reports_dir) / f"replay_{_stamp()}",
                      f"Sentinel replay of live pipeline, last {args.days} days")
    _print_metrics(res.metrics, res.benchmark)
    print(f"\nreplay audit trail -> {replay_db.path} (open it with the dashboard) | report -> {out}")


# ---------------------------------------------------------------------------------------------- live
def cmd_live(cfg: Config, args) -> None:
    from sentinel.live.runner import run_live
    asyncio.run(run_live(cfg, Secrets.from_env(), confirm_live=args.confirm_live))


def _broker(cfg: Config):
    from sentinel.execution.broker import AlpacaBroker
    key, secret = Secrets.from_env().require_alpaca()
    return AlpacaBroker(key, secret, paper=cfg.mode == "paper")


def cmd_status(cfg: Config, args) -> None:
    async def go():
        b = _broker(cfg)
        a, ps, c = await b.get_account(), await b.get_positions(), await b.get_clock()
        print(f"{'PAPER' if cfg.mode == 'paper' else 'LIVE'} account: equity {a.equity:,.2f}  cash {a.cash:,.2f}  "
              f"buying power {a.buying_power:,.2f}  day trades {a.daytrade_count}")
        print(f"market {'OPEN' if c.is_open else 'closed'}; next open {c.next_open}, next close {c.next_close}")
        for p in ps.values():
            print(f"  {p.symbol:6s} {p.qty:>8.0f} @ {p.avg_price:,.2f}  value {p.market_value:,.2f}")
    asyncio.run(go())
    store = Store(cfg.data.db_path)
    ev = store.read_table("risk_events", 10)
    if not ev.empty:
        print("\nrecent risk events:\n" + ev.to_string(index=False))


def cmd_gold_plan(cfg: Config, args) -> None:
    """Run the daily gold plan once against Alpaca (GLD). Schedule it daily after the US close."""
    from alpaca.trading.client import TradingClient

    from sentinel.gold.plan_bot import run
    from sentinel.live.runner import check_live_allowed

    secrets = Secrets.from_env()
    paper = check_live_allowed(cfg, secrets, args.confirm_live)
    key, secret = secrets.require_alpaca()
    plan, intent = run(TradingClient(key, secret, paper=paper), risk=args.risk, dry_run=args.dry_run)
    mode = ("PAPER" if paper else "LIVE") + (", DRY RUN" if args.dry_run else "")
    print(f"\nGold plan ({mode}): {plan.action}: {plan.headline}")
    if plan.entry:
        print(f"  entry ~{plan.entry:,.2f}  stop {plan.stop:,.2f}")
    if plan.trigger_above:
        print(f"  buy trigger: daily close above {plan.trigger_above:,.2f}")
    print(f"  action taken: {intent.kind} {intent.qty or ''} {intent.note}")


def cmd_flatten(cfg: Config, args) -> None:
    async def go():
        b = _broker(cfg)
        ps = await b.get_positions()
        for s in ps:
            if s in cfg.symbols or args.all:
                await b.close_position(s, "manual flatten")
                print(f"closed {s}")
    asyncio.run(go())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="sentinel", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-c", "--config", default="config/config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--synthetic", action="store_true", help="generate offline demo data instead of calling Alpaca")
    f.add_argument("--days", type=int, default=None)
    t = sub.add_parser("train")
    t.add_argument("--holdout-days", type=int, default=0,
                   help="leave the last N calendar days out of training so `replay` is out-of-sample")
    sub.add_parser("backtest")
    r = sub.add_parser("replay")
    r.add_argument("--days", type=int, default=10, help="calendar days at the end of the stored history")
    lv = sub.add_parser("live")
    lv.add_argument("--confirm-live", action="store_true")
    sub.add_parser("status")
    gp = sub.add_parser("gold-plan", help="execute the daily gold plan on Alpaca (GLD); run after the US close")
    gp.add_argument("--risk", type=float, default=0.01, help="fraction of equity lost if the initial stop is hit")
    gp.add_argument("--dry-run", action="store_true", help="show what would be done without sending orders")
    gp.add_argument("--confirm-live", action="store_true")
    fl = sub.add_parser("flatten")
    fl.add_argument("--all", action="store_true", help="also close positions outside the configured symbols")
    args = ap.parse_args(argv)

    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    handlers = {"fetch": cmd_fetch, "train": cmd_train, "backtest": cmd_backtest, "replay": cmd_replay,
                "live": cmd_live, "status": cmd_status, "flatten": cmd_flatten, "gold-plan": cmd_gold_plan}
    try:
        handlers[args.cmd](cfg, args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
