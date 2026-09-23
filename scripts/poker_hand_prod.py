#!/usr/bin/env python3
import json, sys
from pathlib import Path
roots = [Path("../JuniorPoker").resolve(), Path.home() / "JuniorCloud" / "JuniorPoker"]
llm = [Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"]
for p in roots:
    if (p / "juniorpoker" / "hand.py").is_file():
        sys.path.insert(0, str(p)); break
for p in llm:
    if (p / "ports" / "receipt_cache.py").is_file():
        sys.path.insert(0, str(p)); break
from juniorpoker.hand import Hand
print(json.dumps(Hand().start(), indent=2, default=str))
