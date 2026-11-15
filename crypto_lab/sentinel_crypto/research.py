"""Predeclared nested walk-forward selection with a final, frozen 180-day test.

All attempted candidates are exported. An evaluated protocol cannot overwrite
its sealed report. New research requires a new explicit version and new future data.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import ROOT, atomic_json, digest, load_protocol
from .data import read_history
from .execution import simulate
from .features import features, labels, rules, volatility_scale
from .metrics import bootstrap_mean, combine, performance, prediction_metrics
from .models import FAMILIES, THRESHOLDS, fit


def candidates(px, f, y, cfg, boundary):
    pos = rules(px, f, cfg)
    models = {}
    scale = volatility_scale(f, cfg)
    # Predict only current/future evaluation rows, never use fitted values to score a candidate.
    forecast_mask = f.index >= boundary - pd.Timedelta(hours=4)
    for family in FAMILIES:
        model = fit(family, f, y, cfg, boundary)
        p = pd.Series(np.nan, index=f.index)
        p.loc[forecast_mask] = model.predict(f.loc[forecast_mask])
        models[family] = (model, p)
        for threshold in THRESHOLDS:
            pos[f"{family}@{threshold:.2f}"] = (p >= threshold).astype(float) * scale
    return pos, models


def select(px, pos, cfg, start, end):
    trials = []
    for name in pos:
        path, _, account = simulate(px, pos[name], cfg, start=start, end=end)
        stress, _, _ = simulate(px, pos[name], cfg, start=start, end=end, multiplier=cfg["stress_multiplier"])
        m = performance(path, trades=account.closed)
        sm = performance(stress)
        eligible = (name != "cash" and m["round_trips"] >= cfg["min_validation_trades"]
                    and m["sharpe"] > 0.5 and sm["total_return"] > 0
                    and m["max_drawdown"] > -cfg["drawdown_limit"])
        # Frozen objective: stress Sharpe with a small drawdown penalty. Cash is a real option.
        score = sm["sharpe"] + 2 * m["max_drawdown"] if eligible else -1e9
        trials.append({"candidate": name, "eligible": eligible, "score": score, **m,
                       "stress_return": sm["total_return"], "stress_sharpe": sm["sharpe"]})
    valid = [r for r in trials if r["eligible"]]
    winner = max(valid, key=lambda r: (r["score"], r["candidate"]))["candidate"] if valid else "cash"
    return winner, pd.DataFrame(trials)


def policy_for_boundary(px, cfg, boundary):
    """Select on a trailing validation block and refit the chosen family before deployment."""
    f, y = features(px), labels(px, cfg["horizon_bars"])
    val_start = boundary - pd.Timedelta(days=cfg["validation_days"])
    pos, _ = candidates(px, f, y, cfg, val_start)
    winner, trials = select(px, pos, cfg, val_start, boundary)
    model = fit(winner.split("@")[0], f, y, cfg, boundary) if "@" in winner else None
    policy = {"candidate": winner, "model": model, "boundary": str(boundary), "protocol_sha256": digest(cfg)}
    return policy, trials


def policy_positions(px, policy, cfg):
    f = features(px)
    candidate = policy["candidate"]
    if "@" not in candidate:
        return rules(px, f, cfg)[candidate], None
    _, threshold = candidate.split("@")
    model = policy["model"]
    prob = pd.Series(model.predict(f), index=f.index)
    return (prob >= float(threshold)).astype(float) * volatility_scale(f, cfg), prob


def benchmark(px, cfg, start, end, risk_matched):
    f = features(px)
    if risk_matched:
        decision = volatility_scale(f, cfg)
        settings = cfg
    else:
        decision = pd.Series(1.0, index=px.index)
        settings = {**cfg, "max_sleeve_exposure": 1.0}
    return simulate(px, decision, settings, start=start, end=end, risk=False)[0]


def evaluate(root: Path = ROOT) -> dict:
    cfg = load_protocol(root)
    dest = root / "reports" / cfg["version"]
    if (dest / "summary.json").exists():
        report = json.loads((dest / "summary.json").read_text())
        if report["protocol_sha256"] != digest(cfg):
            raise ValueError("Sealed protocol changed. Do not retune against the same test period.")
        print("Using sealed results; no test-set retuning", flush=True)
        return report
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "protocol_locked.json", {"protocol": cfg, "sha256": digest(cfg),
                "locked_utc": datetime.now(timezone.utc).isoformat()})
    histories = {sym: read_history(sym, root) for sym in cfg["symbols"]}
    end = pd.Timestamp(cfg["data_cutoff"])
    holdout_start = end - pd.Timedelta(days=cfg["holdout_days"])
    fold_rows, trial_rows, predictions = [], [], []
    development, holdout, stressed, bh, vm = {}, {}, {}, {}, {}
    total_trades, holdout_trades = 0, 0
    for sym, px in histories.items():
        print(f"Researching {sym}: {len(px)} bars", flush=True)
        f, y = features(px), labels(px, cfg["horizon_bars"])
        first_test = px.index[cfg["warmup_bars"]] + pd.Timedelta(days=cfg["train_days"] + cfg["validation_days"])
        boundary = first_test
        dev_parts, capital = [], 1.0
        fold = 0
        while boundary < holdout_start:
            fold_end = min(boundary + pd.Timedelta(days=cfg["test_days"]), holdout_start)
            if fold_end - boundary < pd.Timedelta(days=15):
                break
            val_start = boundary - pd.Timedelta(days=cfg["validation_days"])
            pos, _ = candidates(px.loc[px.index < fold_end], f.loc[f.index < fold_end],
                                 y, cfg, val_start)
            winner, trials = select(px, pos, cfg, val_start, boundary)
            trials["symbol"], trials["fold"], trials["phase"] = sym, fold, "development_validation"
            trial_rows.append(trials)
            policy = {"candidate": winner, "model": fit(winner.split("@")[0], f, y, cfg, boundary)
                      if "@" in winner else None, "boundary": str(boundary)}
            decision, prob = policy_positions(px.loc[px.index < fold_end], policy, cfg)
            path, _, acc = simulate(px, decision, cfg, start=boundary, end=fold_end)
            dev_parts.append(path.assign(equity=path.equity * capital))
            capital *= float(path.equity.iloc[-1])
            total_trades += acc.closed
            fold_rows.append({"symbol": sym, "fold": fold, "start": str(boundary), "end": str(fold_end),
                              "candidate": winner, **performance(path, trades=acc.closed)})
            print(f"  fold {fold}: {winner}, net {path.equity.iloc[-1]-1:+.2%}", flush=True)
            boundary, fold = fold_end, fold + 1
        development[sym] = pd.concat(dev_parts)
        # Final six months: one selection before the boundary, no reselection/refits inside the test.
        policy, trials = policy_for_boundary(px.loc[px.index < holdout_start], cfg, holdout_start)
        trials["symbol"], trials["fold"], trials["phase"] = sym, "holdout", "final_validation"
        trial_rows.append(trials)
        joblib.dump(policy, dest / f"{sym}-frozen-policy.joblib")
        decision, prob = policy_positions(px, policy, cfg)
        path, fills, acc = simulate(px, decision, cfg, start=holdout_start, end=end)
        holdout[sym] = path
        stressed[sym] = simulate(px, decision, cfg, start=holdout_start, end=end,
                                 multiplier=cfg["stress_multiplier"])[0]
        bh[sym] = benchmark(px, cfg, holdout_start, end, False)
        vm[sym] = benchmark(px, cfg, holdout_start, end, True)
        holdout_trades += acc.closed
        pd.DataFrame(fills, columns=["timestamp", "reason", "side", "units", "price", "fee", "target", "equity"]).to_csv(dest / f"{sym}-holdout-fills.csv", index=False)
        fold_rows.append({"symbol": sym, "fold": "holdout", "start": str(holdout_start), "end": str(end),
                          "candidate": policy["candidate"], **performance(path, trades=acc.closed)})
        # Evaluate every model on frozen test data as diagnostics, never choose by these metrics.
        for family in FAMILIES:
            model = fit(family, f, y, cfg, holdout_start)
            mask = f.index >= holdout_start
            probs = pd.Series(model.predict(f.loc[mask]), index=f.index[mask])
            prediction = prediction_metrics(y.loc[mask], probs, cfg["horizon_bars"], model.majority, model.base_rate)
            predictions.append({"symbol": sym, "family": family, **prediction})
            pd.DataFrame({"prob": probs, "forward_return": y.reindex(probs.index)}).to_csv(dest / f"{sym}-{family}-holdout-predictions.csv")
        print(f"  sealed holdout: {policy['candidate']}, {path.equity.iloc[-1]-1:+.2%}", flush=True)
    portfolio = combine(holdout)
    dev_portfolio = combine(development)
    stress_portfolio, bh_portfolio, vm_portfolio = combine(stressed), combine(bh), combine(vm)
    for name, path in [("holdout", portfolio), ("development", dev_portfolio), ("stress", stress_portfolio),
                       ("buy_hold", bh_portfolio), ("vol_matched", vm_portfolio)]:
        path.to_csv(dest / f"{name}-equity.csv")
    daily = (1 + portfolio["return"]).resample("1D").prod() - 1
    daily_vm = (1 + vm_portfolio["return"]).resample("1D").prod() - 1
    ci = bootstrap_mean(daily, seed=cfg["seed"])
    excess_ci = bootstrap_mean(daily - daily_vm, seed=cfg["seed"])
    hm = performance(portfolio, trades=holdout_trades)
    sm = performance(stress_portfolio)
    vm_m = performance(vm_portfolio)
    checks = {"holdout_net_positive": hm["total_return"] > 0,
              "doubled_costs_positive": sm["total_return"] > 0,
              "beats_vol_matched": hm["total_return"] > vm_m["total_return"],
              "excess_ci_lower_positive": excess_ci["ci95_daily_mean"][0] > 0,
              "at_least_30_holdout_round_trips": holdout_trades >= 30,
              "at_least_60_forward_days_and_30_forward_trades": False}
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(cfg),
        "protocol": cfg, "holdout_start": str(holdout_start), "holdout_end": str(end),
        "data_quality": json.loads((root / "data" / "manifest.json").read_text())["quality"],
        "candidate_count_per_selection": 20, "selection_events": len(trial_rows),
        "candidate_evaluations": sum(len(x) for x in trial_rows),
        "development": performance(dev_portfolio, trades=total_trades), "holdout": hm,
        "holdout_stress": sm, "buy_hold": performance(bh_portfolio), "vol_matched": vm_m,
        "daily_return_uncertainty": ci, "excess_daily_return_uncertainty": excess_ci,
        "prediction_metrics": predictions, "checks": checks,
        "status": "RESEARCH_ONLY_UNPROVEN", "live_money_supported": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "limitations": ["Three surviving, liquid coins selected today: survivorship and universe selection bias.",
          "USDT spot bars from one venue; no futures, borrowing, funding or leverage.",
          "4h bar fills approximate next-open execution; forward paper uses subsequently observed bid/ask.",
          "Circuit breakers operate at observed prices and cannot cap gaps or polling losses.",
          "Block-bootstrap intervals are descriptive and do not remove multiple-testing bias.",
          "Historical results are simulations; the future paper track record starts at worker startup.",
          "News, on-chain, order-book history and macro releases are not in the predictive model."]}
    pd.concat(trial_rows).to_csv(dest / "all-validation-trials.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(dest / "fold-results.csv", index=False)
    pd.DataFrame(predictions).to_csv(dest / "prediction-metrics.csv", index=False)
    atomic_json(dest / "summary.json", summary)
    atomic_json(root / "reports" / "latest.json", {"version": cfg["version"], "summary": str(dest / "summary.json")})
    print(json.dumps({"holdout": hm, "checks": checks}, indent=2), flush=True)
    return summary


def prepare_paper(root: Path = ROOT) -> dict:
    """Current training may use now-past test data; sealed historical results stay immutable.

    This is a new prospective experiment. Its future performance is the test.
    """
    cfg = load_protocol(root)
    boundary = pd.Timestamp(cfg["data_cutoff"])
    folder = root / "runtime" / "policies"
    folder.mkdir(parents=True, exist_ok=True)
    current = {}
    for sym in cfg["symbols"]:
        px = read_history(sym, root)
        policy, trials = policy_for_boundary(px, cfg, boundary)
        f, y = features(px), labels(px, cfg["horizon_bars"])
        policy["all_models"] = {family: fit(family, f, y, cfg, boundary) for family in FAMILIES}
        joblib.dump(policy, folder / f"{sym}.joblib")
        trials.to_csv(folder / f"{sym}-validation.csv", index=False)
        current[sym] = {"candidate": policy["candidate"], "trained_before": str(boundary)}
    atomic_json(folder / "manifest.json", {"protocol_sha256": digest(cfg), "policies": current,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Fixed policies. No automatic tuning against the sealed holdout."})
    return current
