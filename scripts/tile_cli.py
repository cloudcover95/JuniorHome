"""End-user CLI for Home tiles. Local only."""
from __future__ import annotations
import argparse
from pathlib import Path

TILES = {
    "full": Path.home() / ".juniorhome" / "os" / "home_full_tile.json",
    "web3": Path.home() / ".juniorhome" / "os" / "home_web3_tile.json",
}

def show(name: str) -> None:
    p = TILES.get(name)
    if not p or not p.exists():
        print(f"no {name} tile yet")
        return
    print(p.read_text(encoding="utf-8"))

def main() -> None:
    ap = argparse.ArgumentParser(description="JuniorHome inference tiles")
    ap.add_argument("--show", choices=["full", "web3"], default="web3")
    args = ap.parse_args()
    show(args.show)

if __name__ == "__main__":
    main()
