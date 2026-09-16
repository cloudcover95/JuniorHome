#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "tree_dense.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.tree_dense import add, grow
if len(sys.argv) > 1:
    print(json.dumps(add(" ".join(sys.argv[1:])), indent=2))
else:
    print(json.dumps(grow(), indent=2))
