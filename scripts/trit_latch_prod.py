#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "trit_latch.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.trit_latch import bench
print(json.dumps(bench(" ".join(sys.argv[1:]) or "buy oats"), indent=2))
