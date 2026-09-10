#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from packs.junior_theory_cu.charter import CHARTER
from packs.junior_theory_cu.fund import Fund

ROOT = Path.home() / ".juniorhome" / "theory_cu"


def fund() -> Fund:
    return Fund(ROOT)


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "help"
    if cmd == "init":
        fund()
        print(CHARTER)
        return 0
    if cmd == "join":
        print(json.dumps(fund().join(argv[2]).__dict__))
        return 0
    if cmd == "propose":
        print(json.dumps(fund().propose(argv[2], " ".join(argv[3:])).__dict__))
        return 0
    if cmd == "vote":
        print(json.dumps(fund().vote(argv[2], int(argv[3]), argv[4]).__dict__))
        return 0
    if cmd == "board":
        evs = [e.__dict__ for e in fund().forum.events[-20:]]
        print(json.dumps(evs, indent=2))
        return 0
    if cmd == "bundle":
        print(json.dumps(fund().forum.export_bundle(), indent=2)[:500])
        return 0
    print("init|join|propose|vote|board|bundle")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
