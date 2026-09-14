#!/usr/bin/env python3
"""Gaia spine via sibling JuniorLLM. Fallback portrait-only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gaia.py").is_file():
        sys.path.insert(0, str(p))
        break
try:
    from ports.gaia import spine

    who = {"name": sys.argv[1] if len(sys.argv) > 1 else "Gaia", "pronouns": sys.argv[2] if len(sys.argv) > 2 else "they", "voice": "local", "mesh": "omega-stub"}
    print(json.dumps(spine(" ".join(sys.argv[3:]) or "gaia home clock", who), indent=2))
except Exception as e:
    print(json.dumps({"port": "JuniorGaia", "ok": False, "error": type(e).__name__, "download": False, "ue5_launch": False}, indent=2))
