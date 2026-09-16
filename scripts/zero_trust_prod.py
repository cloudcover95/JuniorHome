#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "zero_trust.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.zero_trust import metric
print(json.dumps(metric(" ".join(sys.argv[1:]) or "home dash"), indent=2))
