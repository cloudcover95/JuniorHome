#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorPoker").resolve(), Path.home() / "JuniorCloud" / "JuniorPoker"):
    if (p / "juniorpoker" / "table.py").is_file():
        sys.path.insert(0, str(p)); break
from juniorpoker.table import Table
t = Table(seats=6, decks=2, seed=1)
t.deal_hole()
t.flop()
print(json.dumps({"scene": t.scene(), "peek0_before": t.peek(0), "rub0": t.rub(0), "peek0_after": t.peek(0)}, indent=2))
