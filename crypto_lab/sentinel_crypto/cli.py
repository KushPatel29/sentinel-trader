from __future__ import annotations
import argparse
import json

from .config import ROOT, load_protocol


def main():
    parser = argparse.ArgumentParser(description="Sentinel crypto research and forward paper simulation")
    parser.add_argument("command", choices=["fetch", "research", "prepare-paper", "paper", "status", "stop",
                                           "focus-fetch", "focus-research", "focus-prepare", "focus-watch", "focus-status", "focus-stop", "import-broker", "accuracy-research"])
    parser.add_argument("--once", action="store_true", help="Run one paper observation cycle")
    parser.add_argument("--file", help="Broker CSV file with explicit bid/ask columns and UTC offsets")
    parser.add_argument("--symbol", help="XAUUSD or USOIL/WTIUSD for a broker import")
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
    elif args.command == "focus-fetch":
        from .focus_data import fetch_focus, fetch_oil_reference
        fetch_focus()
        fetch_oil_reference()
    elif args.command == "focus-research":
        from .focus_research import evaluate_focus
        evaluate_focus()
    elif args.command == "accuracy-research":
        from .accuracy import evaluate_accuracy
        evaluate_accuracy()
    elif args.command == "focus-prepare":
        from .focus_watch import prepare
        print(json.dumps(prepare(), indent=2))
    elif args.command == "focus-watch":
        from .focus_watch import run
        run(once=args.once)
    elif args.command == "focus-status":
        path = ROOT / "runtime" / "focus" / "status.json"
        print(path.read_text() if path.exists() else "Focus observer has not started")
    elif args.command == "focus-stop":
        folder = ROOT / "runtime" / "focus"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "STOP").touch()
        print("Focus observer stop requested. This observer has no positions or real orders.")
    elif args.command == "import-broker":
        from pathlib import Path
        from .focus_data import import_broker_csv
        if not args.file or not args.symbol:
            parser.error("import-broker requires --file and --symbol")
        print(import_broker_csv(Path(args.file), args.symbol))


if __name__ == "__main__":
    main()
