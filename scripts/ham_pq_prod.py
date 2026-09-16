#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "ham_pq.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.ham_pq import tool
a = sys.argv[1] if len(sys.argv) > 1 else "buy oats"
b = sys.argv[2] if len(sys.argv) > 2 else None
print(json.dumps(tool(a, b), indent=2))
