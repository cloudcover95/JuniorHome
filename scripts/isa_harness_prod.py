#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "isa_harness.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.isa_harness import harness
print(json.dumps(harness(), indent=2))
