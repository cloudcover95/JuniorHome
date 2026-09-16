#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "receipt_cache.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.receipt_cache import issue
a = sys.argv[1] if len(sys.argv) > 1 else "member share vault"
b = sys.argv[2] if len(sys.argv) > 2 else None
print(json.dumps(issue(a, b), indent=2, default=str))
