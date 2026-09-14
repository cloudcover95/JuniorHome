#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "ship_trit.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.ship_trit import ship
print(json.dumps(ship(" ".join(sys.argv[1:]) or "home dash"), indent=2, default=str))
