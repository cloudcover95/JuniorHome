#!/usr/bin/env python3
import json, sys
from pathlib import Path
roots = [Path("../JuniorPoker").resolve(), Path.home() / "JuniorCloud" / "JuniorPoker"]
llm = [Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"]
for p in roots:
    if (p / "juniorpoker" / "trit_felt.py").is_file():
        sys.path.insert(0, str(p)); break
for p in llm:
    if (p / "ports" / "gaia_proto.py").is_file():
        sys.path.insert(0, str(p)); break
from juniorpoker.table import Table
from juniorpoker.trit_felt import pack
t = Table(seats=6, decks=2, seed=1)
t.deal_hole(); t.flop()
print(json.dumps(pack(t), indent=2, default=str))
