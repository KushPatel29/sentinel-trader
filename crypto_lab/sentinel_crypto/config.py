from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_protocol(root: Path = ROOT) -> dict:
    cfg = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if cfg["interval"] != "4h" or cfg["horizon_bars"] < 1:
        raise ValueError("This protocol requires four-hour bars and a positive horizon")
    if not 0 < cfg["max_sleeve_exposure"] <= 1:
        raise ValueError("Leverage is unsupported")
    if min(cfg["fee_bps"], cfg["slippage_bps"]) < 0:
        raise ValueError("Costs cannot be negative")
    return cfg


def digest(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding="utf-8")
    tmp.replace(path)
