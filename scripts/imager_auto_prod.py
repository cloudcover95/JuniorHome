#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "imager_auto.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.imager_auto import tick
print(json.dumps(tick(" ".join(sys.argv[1:]) or "gaia spine"), indent=2, default=str))
