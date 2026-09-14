#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "fieldcore_spine.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.fieldcore_spine import expand
print(json.dumps(expand(), indent=2))
