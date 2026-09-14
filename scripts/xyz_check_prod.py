#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "xyz_check.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.xyz_check import xyz
print(json.dumps(xyz(" ".join(sys.argv[1:]) or "porch light 127.0.0.1"), indent=2))
