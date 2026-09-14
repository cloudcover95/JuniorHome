#!/usr/bin/env python3
"""Home entry: sibling JuniorLLM ports.home_harness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "home_harness.py").is_file():
        sys.path.insert(0, str(p))
        break
from ports.home_harness import harness

print(json.dumps(harness(sys.argv[1] if len(sys.argv) > 1 else "home clock"), indent=2))
