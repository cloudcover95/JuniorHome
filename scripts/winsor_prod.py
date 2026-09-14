#!/usr/bin/env python3
"""Home wrapper: winsor pack via sibling JuniorLLM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "junior_bitnet" / "winsor.py").is_file():
        sys.path.insert(0, str(p))
        break
from junior_bitnet.winsor import pack

print(json.dumps(pack([0.01, -2.0, 0.4, 9.0, -0.02]), indent=2))
