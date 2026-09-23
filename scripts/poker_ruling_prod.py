#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorPoker").resolve(), Path.home() / "JuniorCloud" / "JuniorPoker"):
    if (p / "juniorpoker" / "ruling.py").is_file():
        sys.path.insert(0, str(p)); break
from juniorpoker.ruling import abc
from juniorpoker.table import Table
t = Table(seats=2, decks=1, seed=11)
t.deal_hole(); t.flop(); t.turn_or_river(); t.turn_or_river()
print(json.dumps(abc(t.hole[0], t.board), indent=2))
