#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "imager_live.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.imager_live import process
print(json.dumps(process(" ".join(sys.argv[1:]) or "gaia spine"), indent=2, default=str))
