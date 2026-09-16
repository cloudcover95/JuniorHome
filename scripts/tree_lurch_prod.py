#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "tree_lurch.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.tree_lurch import lurch, search
cmd, *rest = (sys.argv[1:] or ["lurch"]) + [""]
note = " ".join(rest).strip() or "gaia spine"
if cmd == "search":
    print(json.dumps(search(note), indent=2))
else:
    print(json.dumps(lurch(note), indent=2, default=str))
