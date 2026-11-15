from __future__ import annotations
import argparse
import json

from .config import ROOT, load_protocol


def main():
    parser = argparse.ArgumentParser(description="Sentinel crypto research and forward paper simulation")
    parser.add_argument("command", choices=["fetch", "research", "prepare-paper", "paper", "status", "stop"])
    parser.add_argument("--once", action="store_true", help="Run one paper observation cycle")
    args = parser.parse_args()
    if args.command == "fetch":
        from .data import fetch_history
        fetch_history(load_protocol())
    elif args.command == "research":
        from .research import evaluate
        evaluate()
    elif args.command == "prepare-paper":
        from .research import prepare_paper
        print(json.dumps(prepare_paper(), indent=2))
    elif args.command == "paper":
        from .paper import run
        run(once=args.once)
    elif args.command == "status":
        path = ROOT / "runtime" / "status.json"
        print(path.read_text() if path.exists() else "Worker has not started")
    elif args.command == "stop":
        (ROOT / "runtime").mkdir(exist_ok=True)
        (ROOT / "runtime" / "STOP").touch()
        print("Stop requested. Worker will attempt simulated liquidation with fresh quotes.")


if __name__ == "__main__":
    main()
