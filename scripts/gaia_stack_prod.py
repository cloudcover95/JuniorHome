#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gaia_stack.py").is_file():
        sys.path.insert(0, str(p))
        break
note = " ".join(sys.argv[1:]) or "gaia they terrain"
try:
    from ports.gaia_stack import run

    print(json.dumps(run(note), indent=2))
except Exception as e:
    print(json.dumps({"ok": False, "error": type(e).__name__ + ": " + str(e)[:180], "usgs_fetch": False, "llama_ready": False}, indent=2))
