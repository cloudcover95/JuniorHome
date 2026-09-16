#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "ham_metric.py").is_file():
        sys.path.insert(0, str(p)); break
from ports.ham_metric import metric
a = sys.argv[1] if len(sys.argv) > 1 else "buy oats"
b = sys.argv[2] if len(sys.argv) > 2 else "buy oat"
print(json.dumps(metric(a, b), indent=2))
