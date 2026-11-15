"""Gold, oil and crypto study. V1 is immutable; inspected crypto dates stay exploratory."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from .config import ROOT, atomic_json, digest, load_protocol
from .focus_data import load_focus, read_focus
from .focus_execution import focus_performance, simulate_focus
from .focus_features import build_features, exposure_scale, rule_library, signal_series, target
from .focus_models import choose_forecaster, fit_library
from .metrics import bootstrap_mean


def data_gate(px, spec, start, end):
    section = px[(px.index >= start)&(px.index < end)]
    if section.empty:
        return False, {"reason": "no data in required window"}
    gap = section.index.to_series().diff().dt.total_seconds().div(86400)
    max_gap = float(gap.max()) if len(gap) > 1 else 0.
    if spec["interval"] in ("1h", "1d"):
        expected = len(pd.bdate_range(start.normalize().tz_localize(None), end.normalize().tz_localize(None), inclusive="left"))
        observed = len(section.index[section.index.dayofweek < 5].normalize().unique())
    else:
        expected = int((end-start).total_seconds()/86400)
        observed = len(section.index.normalize().unique())
    coverage = observed/max(expected, 1)
    okay = coverage >= .85 and max_gap <= 7
    return okay, {"day_coverage": coverage, "max_gap_days": max_gap, "required_start": str(start), "required_end": str(end),
                  "actual_start": str(section.index[0]), "actual_last_open": str(section.index[-1]),
                  "reason": "passed" if okay else "insufficient coverage or a gap over seven days; no splicing/imputation"}


def read_context(cfg, root):
    out = {}
    for key, symbol in cfg["context_symbols"].items():
        frame = pd.read_csv(root / "data" / "focus" / "context" / (symbol.replace("^", "")+".csv"), index_col=0, parse_dates=True)
        frame.index = pd.to_datetime(frame.index, utc=True)
        out[key] = frame
    return out


def prediction_summary(frame):
    if frame.empty:
        return {"samples": 0}
    yy = frame.actual_up.astype(int)
    p = frame.probability.clip(1e-6, 1-1e-6)
    pred = p >= frame.threshold
    base = frame.prior_probability
    result = {"samples": len(frame), "accuracy": float(accuracy_score(yy, pred)),
              "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
              "auc": float(roc_auc_score(yy, p)) if yy.nunique() > 1 else None,
              "brier": float(brier_score_loss(yy, p)), "prior_accuracy": float(accuracy_score(yy, base >= .5)),
              "prior_brier": float(brier_score_loss(yy, base)),
              "price_only_accuracy": float(accuracy_score(yy, frame.price_probability >= frame.price_threshold)),
              "price_only_brier": float(brier_score_loss(yy, frame.price_probability)),
              "mean_predicted": float(p.mean()), "observed_up_rate": float(yy.mean())}
    result["brier_skill_vs_prior"] = 1-result["brier"]/result["prior_brier"]
    acc_diff = ((pred == yy).astype(float)-((base >= .5) == yy).astype(float))
    daily = pd.Series(acc_diff.to_numpy(), index=pd.to_datetime(frame.timestamp, utc=True).dt.floor("D")).groupby(level=0).mean()
    result["accuracy_difference_vs_prior_ci"] = bootstrap_mean(daily, block=14)
    loss_diff = pd.Series(((yy-base)**2-(yy-p)**2).to_numpy(), index=daily_index(frame)).groupby(level=0).mean()
    result["brier_improvement_ci"] = bootstrap_mean(loss_diff, block=14)
    if "v1_probability" in frame and frame.v1_probability.notna().all():
        v1 = frame.v1_probability
        result["v1_ensemble_accuracy"] = float(accuracy_score(yy, v1 >= .5))
        result["v1_ensemble_brier"] = float(brier_score_loss(yy, v1))
        v1diff = pd.Series(((yy-v1)**2-(yy-p)**2).to_numpy(), index=daily_index(frame)).groupby(level=0).mean()
        result["brier_improvement_vs_v1_ci"] = bootstrap_mean(v1diff, block=14)
    buckets = pd.cut(p, [0, .4, .45, .5, .55, .6, 1], include_lowest=True)
    temp = pd.DataFrame({"p": p, "y": yy, "bucket": buckets.astype(str)})
    result["calibration"] = temp.groupby("bucket", observed=True).agg(predicted=("p", "mean"), actual_up=("y", "mean"), samples=("y", "size")).reset_index().to_dict("records")
    confident = (p-.5).abs() >= .07
    result["confident_coverage"] = float(confident.mean())
    result["confident_accuracy"] = float(accuracy_score(yy[confident], p[confident] >= .5)) if confident.any() else None
    return result


def daily_index(frame):
    return pd.to_datetime(frame.timestamp, utc=True).dt.floor("D").to_numpy()


def nonoverlapping(frame):
    rows, previous_end = [], None
    for row in frame.sort_values("timestamp").itertuples(index=False):
        ts, end = pd.Timestamp(row.timestamp), pd.Timestamp(row.label_end)
        if previous_end is None or ts >= previous_end:
            rows.append(row._asdict())
            previous_end = end
    return pd.DataFrame(rows, columns=frame.columns)


def policy_candidates(px, f, probabilities, cfg, spec):
    p = rule_library(px, f, spec, cfg)
    scale = exposure_scale(f, cfg)
    for name in probabilities:
        for threshold in cfg["entry_thresholds"]:
            directions = cfg["directions"] if spec["source"] == "histdata" else ["long_only"]
            for direction in directions:
                key = f"{name}@{threshold:.2f}@{direction}"
                p[key] = signal_series(probabilities[name], threshold, direction, spec["horizon_bars"])*scale
    p.loc[f.vol_regime > 2.5] = 0.
    return p


def select_policy(px, positions, cfg, spec, start, end):
    trials = []
    for name in positions:
        path, _, stats = simulate_focus(px, positions[name], cfg, spec, start, end, multiplier=2.)
        m = focus_performance(path, spec, stats)
        daily = (1+path["return"]).resample("1D").prod()-1
        thirds = np.array_split(daily.to_numpy(), 3)
        profitable_parts = sum(float(np.prod(1+part)-1) > 0 for part in thirds if len(part))
        eligible = (name != "cash" and stats["round_trips"] >= cfg["min_validation_trades"]
                    and m["total_return"] > 0 and m["sharpe"] > .5 and profitable_parts >= 2
                    and m["max_drawdown"] > -cfg["drawdown_limit"] and stats["overnight_cfd_carries"] == 0)
        score = m["sharpe"]+2*m["max_drawdown"] if eligible else -1e9
        trials.append({"candidate": name, "eligible": eligible, "stress_score": score, "profitable_validation_thirds": profitable_parts, **m})
    valid = [r for r in trials if r["eligible"]]
    winner = max(valid, key=lambda r: (r["stress_score"], r["candidate"]))["candidate"] if valid else "cash"
    return winner, pd.DataFrame(trials)


def v1_ensemble(px, start, end, root):
    from .features import features, labels
    from .models import FAMILIES, fit
    cfg = load_protocol(root)
    f = features(px)
    y = labels(px, cfg["horizon_bars"])
    models = [fit(family, f, y, cfg, start) for family in FAMILIES]
    mask = (f.index >= start)&(f.index < end)
    return pd.Series(np.mean([model.predict(f.loc[mask]) for model in models], axis=0), index=f.index[mask])


def evaluate_focus(root: Path = ROOT):
    cfg = load_focus(root)
    dest = root / "reports" / cfg["version"]
    if (dest / "summary.json").exists():
        summary = json.loads((dest / "summary.json").read_text())
        if summary["protocol_sha256"] != digest(cfg):
            raise ValueError("Focus protocol changed after results were sealed")
        return summary
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "protocol_locked.json", {"protocol": cfg, "sha256": digest(cfg), "locked_utc": datetime.now(timezone.utc).isoformat()})
    contexts = read_context(cfg, root)
    manifest = json.loads((root / "data" / "focus" / "manifest.json").read_text())
    assets, all_forecasts, validation_tables, trade_tables, fold_tables = {}, [], [], [], []
    final_start, cutoff = pd.Timestamp(cfg["final_start"]), pd.Timestamp(cfg["cutoff"])
    for symbol, spec in cfg["assets"].items():
        print(f"Focus research {symbol}", flush=True)
        px = read_focus(symbol, root)
        f, groups = build_features(px, spec, contexts)
        y, label_end = target(px, spec)
        history_start = px.index[0]+pd.Timedelta(days=cfg["train_days"]+cfg["validation_days"]+30)
        first = max(pd.Timestamp(cfg["first_development_test"]), history_start.ceil("D"))
        boundaries = []
        at = first
        while at < final_start:
            until = min(at+pd.Timedelta(days=cfg["test_days"]), final_start)
            if until-at >= pd.Timedelta(days=45):
                boundaries.append((at, until, "development"))
            at = until
        boundaries.append((final_start, cutoff, "final_retrospective"))
        asset_rows, equity_parts, stress_parts, base_parts, blocked = [], [], [], [], []
        capitals = {"policy": 1., "stress": 1., "baseline": 1.}
        final_result, final_policy = None, None
        for number, (start, end, phase) in enumerate(boundaries):
            validation_start = start-pd.Timedelta(days=cfg["validation_days"])
            okay, gate = data_gate(px, spec, validation_start-pd.Timedelta(days=cfg["train_days"]), end)
            if not okay:
                blocked.append({"phase": phase, "start": str(start), "end": str(end), **gate})
                print(f"  BLOCKED {phase}: coverage {gate.get('day_coverage',0):.1%}, gap {gate.get('max_gap_days',0):.1f} days", flush=True)
                continue
            # Candidate training ends before validation. No fitting on the score/test window.
            try:
                models = fit_library(f, y, label_end, groups, cfg, validation_start)
            except ValueError as exc:
                blocked.append({"phase": phase, "start": str(start), "reason": str(exc)})
                continue
            score_mask = (f.index >= validation_start-pd.Timedelta(spec["interval"]))&(f.index < end)
            probability = pd.DataFrame({name: model.predict(f.loc[score_mask]) for name, model in models.items()}, index=f.index[score_mask])
            # Validation labels must be known before selection at the test boundary.
            safe_y = y.where(label_end < start)
            chosen, scores = choose_forecaster(models, probability, safe_y, validation_start, start)
            price_names = {k: v for k, v in models.items() if k.endswith(":price")}
            price_chosen, _ = choose_forecaster(price_names, probability[list(price_names)], safe_y, validation_start, start)
            scores["symbol"], scores["fold"], scores["phase"] = symbol, number, phase
            validation_tables.append(scores)
            if not spec.get("direction_only"):
                # Use the exact same pre-validation-fitted models for validation and test.
                # Avoid refitting a selected model whose validation calibration would change.
                positions = policy_candidates(px, f, probability.reindex(f.index), cfg, spec)
                policy_name, trials = select_policy(px, positions, cfg, spec, validation_start, start)
                trials["symbol"], trials["fold"], trials["phase"] = symbol, number, phase
                trade_tables.append(trials)
                policy_path, fills, stats = simulate_focus(px, positions[policy_name], cfg, spec, start, end)
                stress_path, _, _ = simulate_focus(px, positions[policy_name], cfg, spec, start, end, multiplier=4.)
                baseline = rule_library(px, f, spec, cfg)["trend_long"] if spec["intraday_only"] else exposure_scale(f, cfg)
                baseline_path, _, baseline_stats = simulate_focus(px, baseline, cfg, spec, start, end, risks=False)
                perf = focus_performance(policy_path, spec, stats)
                fold_tables.append({"symbol": symbol, "fold": number, "phase": phase, "start": str(start), "end": str(end),
                                    "forecaster": chosen, "policy": policy_name, **perf})
                if phase == "final_retrospective":
                    final_result = {"policy": policy_name, "metrics": perf,
                                    "stress_4x": focus_performance(stress_path, spec),
                                    "baseline": focus_performance(baseline_path, spec, baseline_stats)}
                    policy_path.to_csv(dest / f"{symbol}-final-equity.csv")
                    stress_path.to_csv(dest / f"{symbol}-final-stress-equity.csv")
                    baseline_path.to_csv(dest / f"{symbol}-final-baseline-equity.csv")
                    pd.DataFrame(fills, columns=["timestamp", "reason", "side", "units", "price", "equity"]).to_csv(dest / f"{symbol}-final-fills.csv", index=False)
                for key, path, pieces in [("policy", policy_path, equity_parts), ("stress", stress_path, stress_parts), ("baseline", baseline_path, base_parts)]:
                    pieces.append(path.assign(equity=path.equity*capitals[key]))
                    capitals[key] *= float(path.equity.iloc[-1])
            else:
                policy_name = "DIRECTION_STUDY_ONLY"
            # Truth labels unavailable by the actual test end cannot be scored.
            mask = (f.index >= start)&(f.index < end)&y.notna()&(label_end < end)
            frame = pd.DataFrame({"timestamp": f.index[mask], "label_end": label_end.loc[mask].to_numpy(),
                "actual_up": (y.loc[mask] > 0).astype(int).to_numpy(),
                "probability": probability[chosen].reindex(f.index[mask]).to_numpy(),
                "threshold": models[chosen].threshold, "prior_probability": probability["prior:price"].reindex(f.index[mask]).to_numpy(),
                "price_probability": probability[price_chosen].reindex(f.index[mask]).to_numpy(),
                "price_threshold": models[price_chosen].threshold, "symbol": symbol, "phase": phase,
                "forecaster": chosen, "price_forecaster": price_chosen})
            if spec["source"] == "binance":
                v1 = v1_ensemble(px, start, end, root)
                frame["v1_probability"] = v1.reindex(f.index[mask]).to_numpy()
            frame = nonoverlapping(frame)
            asset_rows.append(frame)
            if phase == "final_retrospective":
                final_policy = {"model": models[chosen], "name": chosen, "policy": policy_name, "spec": spec,
                                "fit_before": str(validation_start), "test_start": str(start), "test_end": str(end)}
                joblib.dump(final_policy, dest / f"{symbol}-retrospective-model.joblib")
            print(f"  {phase} {str(start)[:10]}: predictor {chosen}, policy {policy_name}, n={len(frame)}", flush=True)
        merged = pd.concat(asset_rows, ignore_index=True) if asset_rows else pd.DataFrame()
        final_frame = merged[merged.phase == "final_retrospective"] if len(merged) else merged
        dev_frame = merged[merged.phase == "development"] if len(merged) else merged
        if len(merged):
            merged.to_csv(dest / f"{symbol}-oos-forecasts.csv", index=False)
            all_forecasts.append(merged)
        latest_age = (cutoff-(px.index[-1]+pd.Timedelta(spec["interval"]))).total_seconds()/86400
        assets[symbol] = {"name": spec["name"], "spec": spec, "data": manifest["quality"][symbol],
            "status": "DATA_BLOCKED" if final_policy is None else "DIRECTION_ONLY" if spec.get("direction_only") else "RESEARCH_ONLY",
            "stale_days_at_cutoff": latest_age, "blocked_windows": blocked,
            "development_predictions": prediction_summary(dev_frame), "final_predictions": prediction_summary(final_frame),
            "final_trading": final_result, "latest_model": final_policy["name"] if final_policy else None,
            "execution_ready": False, "best_in_class_proven": False,
            "test_interpretation": "Exploratory: crypto dates were inspected in v1; not a new untouched test." if spec["source"] == "binance" else
                                   "New asset retrospective study. Reference quotes and assumed execution costs; independent broker replay still required."}
    for name, frames in [("all-model-validation", validation_tables), ("all-policy-validation", trade_tables)]:
        pd.concat(frames, ignore_index=True).to_csv(dest / f"{name}.csv", index=False)
    pd.DataFrame(fold_tables).to_csv(dest / "fold-results.csv", index=False)
    summary = {"version": cfg["version"], "protocol_sha256": digest(cfg), "created_utc": datetime.now(timezone.utc).isoformat(),
               "protocol": cfg, "assets": assets, "model_validation_evaluations": sum(len(x) for x in validation_tables),
               "policy_validation_evaluations": sum(len(x) for x in trade_tables),
               "status": "RESEARCH_ONLY_NOT_BEST_IN_CLASS", "broker": "not selected", "live_orders_supported": False,
               "limitations": ["HistData is a reference BID feed without consolidated volume or historical spread.",
                  "WTI CFD data has a 940-day gap; affected windows are blocked rather than filled from another instrument.",
                  "The separately labeled CL=F daily direction study is not USOIL CFD P&L.",
                  "CFD session/short costs are assumptions and not a verified broker contract.",
                  "Context is delayed until next UTC midnight; no revised macro series or historical news is advertised as point-in-time data.",
                  "Prediction selection uses validation Brier score; strategy selection separately requires stress returns and consistency.",
                  "Crypto dates already seen in v1 remain exploratory. No claim of a newly untouched crypto holdout.",
                  "Intervals are descriptive block bootstraps and do not establish multiple-comparison-adjusted superiority."]}
    atomic_json(dest / "summary.json", summary)
    print(json.dumps({s: {"status": x["status"], "predictions": x["final_predictions"].get("accuracy"),
                             "return": x["final_trading"]["metrics"]["total_return"] if x["final_trading"] else None} for s, x in assets.items()}, indent=2), flush=True)
    return summary
