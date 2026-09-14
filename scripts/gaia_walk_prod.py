#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gaia_walk.py").is_file():
        sys.path.insert(0, str(p))
        break
try:
    from ports.gaia_walk import walk

    print(json.dumps(walk(" ".join(sys.argv[1:]) or "flagstaff mountain"), indent=2))
except Exception as e:
    print(json.dumps({"ok": False, "error": type(e).__name__, "scrape": False}, indent=2))
