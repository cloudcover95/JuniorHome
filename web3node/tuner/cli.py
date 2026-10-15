"""CLI for tuner + dyno. Local only. No flash."""
from __future__ import annotations
import argparse, json, sys
from .manager import run
from .ingest import ingest
from .runs import load, compare

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tuner")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("run")
    ing = sub.add_parser("ingest")
    ing.add_argument("profile")
    ing.add_argument("raw")
    cmp = sub.add_parser("compare")
    cmp.add_argument("a", type=int)
    cmp.add_argument("b", type=int)
    args = p.parse_args(argv)
    if args.cmd == "run":
        print(json.dumps(run(), indent=2))
    elif args.cmd == "ingest":
        raw = json.loads(args.raw)
        print(json.dumps(ingest(args.profile, raw), indent=2))
    elif args.cmd == "compare":
        rows = load()
        if args.a >= len(rows) or args.b >= len(rows):
            print(json.dumps({"ok": False, "reason": "index"}))
            return 1
        print(json.dumps(compare(rows[args.a], rows[args.b]), indent=2))
    else:
        p.print_help()
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
