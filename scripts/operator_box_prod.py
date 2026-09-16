#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "operator_box.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.imager_auto import tick
from ports.operator_box import status
from ports.tree_dense import grow
print(json.dumps({"box": status(), "tick": tick("gaia spine"), "tree": grow()}, indent=2, default=str))
