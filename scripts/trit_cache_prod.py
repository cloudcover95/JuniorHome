#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "trit_cache.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.trit_cache import hit, put, replay
args = sys.argv[1:]
if args and args[0] == "replay":
    print(json.dumps(replay(args[1:] or ["buy oats", "buy oat", "bind 0.0.0.0"]), indent=2))
elif args and args[0] == "hit":
    print(json.dumps(hit(" ".join(args[1:]) or "buy oats"), indent=2))
else:
    print(json.dumps(put(" ".join(args) or "buy oats"), indent=2))
