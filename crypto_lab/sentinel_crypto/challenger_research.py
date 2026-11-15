"""Frozen, resumable real-data comparison for trees, LSTM, TFT and a hybrid."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from .challenger_models import fit_challenger, hybrid_weights
from .config import ROOT, atomic_json, digest
from .focus_data import load_focus, read_focus
from .focus_features import build_features, target
from .focus_research import data_gate
from .metrics import bootstrap_mean


def load_challengers(root=ROOT):
    return json.loads((root / "challenger_protocol.json").read_text())


def evaluation_positions(f, y, ends, start, end, spec, reference=None):
    valid = (f.index >= start) & (f.index < end) & y.notna() & (ends < end)
    if reference is not None:
        valid &= f.index.isin(reference)
    positions, next_allowed = [], start
    for i in np.flatnonzero(valid):
        entry = f.index[i] + pd.Timedelta(spec["interval"])
        if entry >= next_allowed:
            positions.append(i)
            next_allowed = ends.iloc[i]
    return np.asarray(positions, dtype=int)


def metrics(frame, name):
    p, y = frame[name].to_numpy(), frame.actual_up.to_numpy()
    daily = (
        pd.Series(
            ((p >= 0.5) == y).astype(float) - ((frame.prior.to_numpy() >= 0.5) == y).astype(float),
            index=pd.to_datetime(frame.timestamp, utc=True).dt.floor("D"),
        )
        .groupby(level=0)
        .mean()
    )
    return {
        "samples": len(y),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "accuracy_gain_vs_prior_pp": float(100 * (accuracy_score(y, p >= 0.5) - accuracy_score(y, frame.prior >= 0.5))),
        "accuracy_gain_vs_prior_ci": bootstrap_mean(daily),
        "mean_probability": float(p.mean()),
        "median_probability": float(np.median(p)),
        "coverage": 1.0,
    }


def fold(symbol, px, spec, boundary, until, cfg, folder):
    key = f"{symbol}-{boundary.date()}"
    output = folder / f"{key}-forecasts.csv"
    if output.exists():
        return pd.read_csv(output)
    f, _ = build_features(px, spec, {})
    y, ends = target(px, spec)
    grid = folder.parents[1] / "accuracy_evidence" / f"{symbol}-aligned-forecasts.csv"
    reference = pd.to_datetime(pd.read_csv(grid, usecols=["timestamp"]).timestamp, utc=True) if grid.exists() else None
    positions = evaluation_positions(f, y, ends, boundary, until, spec, reference)
    if len(positions) < 20:
        raise ValueError("Not enough non-overlapping forecasts")
    frame = pd.DataFrame(
        {
            "timestamp": f.index[positions],
            "actual_up": (y.iloc[positions] > 0).astype(int).to_numpy(),
            "actual_return": y.iloc[positions].to_numpy(),
            "label_end": ends.iloc[positions].to_numpy(),
        }
    )
    audits = []
    for name in cfg["models"]:
        checkpoint = folder / f"{key}-{name}.joblib"
        before = time.monotonic()
        print(f"{symbol} {boundary.date()} {name}: fitting", flush=True)
        if checkpoint.exists():
            model = joblib.load(checkpoint)
        else:
            model = fit_challenger(name, f, y, ends, boundary, cfg)
            joblib.dump(model, checkpoint, compress=3)
        frame[name] = model.predict(f, positions)
        if "prior" not in frame:
            frame["prior"] = model.prior
        audits.append({"symbol": symbol, "model": name, **model.audit, "elapsed_seconds": time.monotonic() - before})
        print(
            f"  {name}: {accuracy_score(frame.actual_up, frame[name] >= 0.5):.2%}, {time.monotonic() - before:.1f}s",
            flush=True,
        )
        atomic_json(
            folder / "progress.json",
            {
                "symbol": symbol,
                "boundary": str(boundary),
                "last_model": name,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    frame.to_csv(output, index=False)
    atomic_json(folder / f"{key}-fit-audits.json", audits)
    return frame


def evaluate(root: Path = ROOT):
    cfg, focus = load_challengers(root), load_focus(root)
    folder = root / "reports" / cfg["version"]
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "protocol_locked.json"
    source_hashes = {
        s: hashlib.sha256((root / "data" / "focus" / f"{s}.csv").read_bytes()).hexdigest() for s in cfg["assets"]
    }
    if lock.exists():
        old = json.loads(lock.read_text())
        if old["sha256"] != digest(cfg) or old["source_sha256"] != source_hashes:
            raise ValueError("Protocol or source data changed after locking this run")
    else:
        atomic_json(
            lock,
            {
                "protocol": cfg,
                "sha256": digest(cfg),
                "source_sha256": source_hashes,
                "locked_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    if (folder / "summary.json").exists():
        return json.loads((folder / "summary.json").read_text())
    columns = ["prior"] + cfg["models"]
    assets, rows = {}, []
    dev_start, final_start, refit, end = map(
        pd.Timestamp, [cfg["development_start"], cfg["final_start"], cfg["refit_boundary"], cfg["end"]]
    )
    for symbol in cfg["assets"]:
        spec, px = focus["assets"][symbol], read_focus(symbol, root)
        okay, gate = data_gate(px, spec, dev_start - pd.Timedelta(days=cfg["train_days"]), end)
        if not okay:
            assets[symbol] = {"status": "DATA_BLOCKED", "gate": gate, "orders_enabled": False}
            atomic_json(folder / "partial_summary.json", assets)
            continue
        dev = fold(symbol, px, spec, dev_start, final_start, cfg, folder)
        weights = hybrid_weights(dev[columns], dev.actual_up, cfg)
        weight_map = dict(zip(columns, weights.tolist()))
        dev["hybrid"] = cfg["hybrid_prior_weight"] * dev.prior + (1 - cfg["hybrid_prior_weight"]) * (
            dev[columns].to_numpy() @ weights
        )
        scores = {name: metrics(dev, name) for name in columns + ["hybrid"]}
        chosen = min(scores, key=lambda n: (scores[n]["brier"], n))
        # Selection and mixture weights are sealed before the later scores exist.
        atomic_json(
            folder / f"{symbol}-selection.json",
            {
                "selected": chosen,
                "weights": weight_map,
                "development": scores,
                "hybrid_development_scores_are_fitted": True,
                "selection_end": str(final_start),
                "protocol_sha256": digest(cfg),
            },
        )
        future = pd.concat(
            [fold(symbol, px, spec, final_start, refit, cfg, folder), fold(symbol, px, spec, refit, end, cfg, folder)],
            ignore_index=True,
        )
        future["hybrid"] = cfg["hybrid_prior_weight"] * future.prior + (1 - cfg["hybrid_prior_weight"]) * (
            future[columns].to_numpy() @ weights
        )
        final = {name: metrics(future, name) for name in columns + ["hybrid"]}
        # Compare released v3 only on the exact intersection of timestamps.
        path = root / "accuracy_evidence" / f"{symbol}-aligned-forecasts.csv"
        matched = {}
        if path.exists():
            old = pd.read_csv(path)
            old_report = json.loads((root / "accuracy_evidence" / "summary.json").read_text())
            profile = old_report["assets"][symbol]["selected_profile"]
            old["timestamp"] = pd.to_datetime(old.timestamp, utc=True)
            same = future.copy()
            same["timestamp"] = pd.to_datetime(same.timestamp, utc=True)
            same = same.merge(
                old[["timestamp", profile]].rename(columns={profile: "released_v3"}), on="timestamp", how="inner"
            )
            if len(same):
                matched = {
                    "samples": len(same),
                    "v3_accuracy": float(accuracy_score(same.actual_up, same.released_v3 >= 0.5)),
                    "v4_selected_accuracy": float(accuracy_score(same.actual_up, same[chosen] >= 0.5)),
                    "v3_brier": float(brier_score_loss(same.actual_up, same.released_v3)),
                    "v4_selected_brier": float(brier_score_loss(same.actual_up, same[chosen])),
                }
        future.to_csv(folder / f"{symbol}-retrospective-forecasts.csv", index=False)
        for name, m in final.items():
            rows.append(
                {
                    "symbol": symbol,
                    "model": name,
                    "selected_on_development": name == chosen,
                    **{k: v for k, v in m.items() if not isinstance(v, dict)},
                }
            )
        assets[symbol] = {
            "status": "DIRECTION_ONLY" if spec.get("direction_only") else "RETROSPECTIVE_ONLY",
            "selected": chosen,
            "weights": weight_map,
            "development": scores,
            "retrospective": final,
            "matched_v3": matched,
            "source_last_open": str(px.index[-1]),
            "orders_enabled": False,
            "execution_ready": False,
        }
        atomic_json(folder / "partial_summary.json", assets)
        pd.DataFrame(rows).to_csv(folder / "all-model-scores.csv", index=False)
    result = {
        "version": cfg["version"],
        "protocol_sha256": digest(cfg),
        "assets": assets,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": cfg["interpretation"],
        "orders_enabled": False,
        "neural_implementation": "torch.nn.LSTM and pytorch_forecasting.TemporalFusionTransformer (binary classification head)",
        "limitations": [
            "Previously inspected history; independent future replication required.",
            "Hybrid weights are fitted on development observations and frozen for later evaluation.",
            "Small CPU neural configurations and bounded training budgets; not an exhaustive architecture search.",
            "Directional accuracy does not establish trading profit after costs.",
            "CFD live testing remains blocked without qualified current broker quotes.",
        ],
    }
    atomic_json(folder / "summary.json", result)
    return result


def prepare_forward(root: Path = ROOT):
    from .public_live import recent_bars
    from .data import validate

    cfg, focus = load_challengers(root), load_focus(root)
    evidence = json.loads((root / "reports" / cfg["version"] / "summary.json").read_text())
    folder = root / "runtime" / "challengers"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "manifest.json").exists():
        raise ValueError("Forward inception already exists; do not silently replace frozen models")
    assets = {}
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        base = read_focus(symbol, root)
        recent, now = recent_bars(symbol)
        px = pd.concat([base, recent])
        px = px[~px.index.duplicated(keep="last")].sort_index()
        validate(px, require_contiguous=True)
        boundary = now.floor("4h")
        f, _ = build_features(px, focus["assets"][symbol], {})
        y, ends = target(px, focus["assets"][symbol])
        members = {}
        for name in cfg["models"]:
            print(f"Prospective fit {symbol} {name} before {boundary}", flush=True)
            members[name] = fit_challenger(name, f, y, ends, boundary, cfg)
        artifact = folder / f"{symbol}.joblib"
        joblib.dump(
            {
                "members": members,
                "weights": evidence["assets"][symbol]["weights"],
                "selected": evidence["assets"][symbol]["selected"],
                "boundary": str(boundary),
                "spec": focus["assets"][symbol],
                "orders_enabled": False,
            },
            artifact,
            compress=3,
        )
        assets[symbol] = {
            "artifact": artifact.name,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "models": cfg["models"] + ["prior", "hybrid"],
            "fit_before": str(boundary),
        }
    manifest = {
        "version": cfg["version"],
        "protocol_sha256": digest(cfg),
        "assets": assets,
        "inception_utc": datetime.now(timezone.utc).isoformat(),
        "orders_enabled": False,
    }
    atomic_json(folder / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_forward() if args.prepare else evaluate(), indent=2))
