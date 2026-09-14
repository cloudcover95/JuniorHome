#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gaia_agent.py").is_file():
        sys.path.insert(0, str(p))
        break

n = 32
args = [a for a in sys.argv[1:] if not a.startswith("--n=")]
for a in sys.argv[1:]:
    if a.startswith("--n="):
        n = int(a.split("=", 1)[1])
note = " ".join(args) or "gaia terrain flagstaff"
out = Path.home() / ".juniorhome" / "gaia_mesh"
try:
    from ports.gaia_agent import run

    print(json.dumps(run(note, out, n), indent=2))
except Exception as e:
    print(json.dumps({"ok": False, "error": type(e).__name__ + ": " + str(e)[:160], "ue5_launch": False}, indent=2))
