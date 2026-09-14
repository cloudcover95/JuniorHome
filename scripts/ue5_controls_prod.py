#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "ue5_api.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.ue5_api import write
print(json.dumps(write(), indent=2))
