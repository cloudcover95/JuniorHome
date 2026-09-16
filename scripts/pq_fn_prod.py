#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "pq_fn.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.pq_fn import fn
print(json.dumps(fn(sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
