#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "tree_zoom.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.tree_zoom import grow, tick
args = sys.argv[1:]
if args:
    print(json.dumps(tick(" ".join(args)), indent=2, default=str))
else:
    print(json.dumps(grow(), indent=2, default=str))
