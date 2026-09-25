#!/usr/bin/env python3
import json, sys
from pathlib import Path
for p in (Path(__file__).resolve().parents[1].parent / "JuniorLLM", Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "trit_gates.py").is_file():
        sys.path.insert(0, str(p))
        break
from ports.trit_gates import apply, status
print(json.dumps({"neg": apply("neg", [1, 0, -1]), "min": apply("min", [1, -1, 1, 1]), "st": status()}, indent=2))
