#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "winsor_brain.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.winsor_brain import tick
print(json.dumps(tick(" ".join(sys.argv[1:]) or "journal field note"), indent=2, default=str))
