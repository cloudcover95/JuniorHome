"""End-user CLI for the Home full tile. Local only."""
from __future__ import annotations
import argparse
from pathlib import Path

TILE = Path.home() / ".juniorhome" / "os" / "home_full_tile.json"

def show() -> None:
    if not TILE.exists():
        print("no tile yet — run scripts/home_full_tile.py")
        return
    print(TILE.read_text(encoding="utf-8"))

def main() -> None:
    ap = argparse.ArgumentParser(description="JuniorHome full inference tile")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if args.show:
        show()
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
