#!/usr/bin/env python3
from __future__ import annotations

import json, sys
from pathlib import Path
for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "markets_view.py").is_file():
        sys.path.insert(0, str(p)); break
orient, scale, args = "landscape", 1.0, []
for a in sys.argv[1:]:
    if a.startswith("--orient="): orient = a.split("=",1)[1]
    elif a.startswith("--scale="): scale = float(a.split("=",1)[1])
    else: args.append(a)
from ports.markets_view import board
print(json.dumps(board(" ".join(args) or "stock tape", orient, scale), indent=2))
