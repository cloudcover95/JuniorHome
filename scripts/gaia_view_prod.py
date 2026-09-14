#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gaia_view.py").is_file():
        sys.path.insert(0, str(p))
        break
orient = "landscape"
scale = 1.0
args = []
for a in sys.argv[1:]:
    if a.startswith("--orient="):
        orient = a.split("=", 1)[1]
    elif a.startswith("--scale="):
        scale = float(a.split("=", 1)[1])
    else:
        args.append(a)
try:
    from ports.gaia_view import view

    print(json.dumps(view(" ".join(args) or "home dash", orient, scale), indent=2))
except Exception as e:
    print(json.dumps({"ok": False, "error": type(e).__name__}, indent=2))
