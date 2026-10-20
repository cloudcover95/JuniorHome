"""End-user CLI for RLVR envelope and handoff. Local only."""
from __future__ import annotations
import argparse
from pathlib import Path

def show() -> None:
    p = Path.home() / ".juniorhome" / "os" / "home_rlvr.json"
    if not p.exists():
        print("no rlvr status yet")
        return
    print(p.read_text(encoding="utf-8"))

def main() -> None:
    ap = argparse.ArgumentParser(description="JuniorHome RLVR status")
    ap.add_argument("--show", action="store_true", help="print latest envelope")
    args = ap.parse_args()
    if args.show:
        show()
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
