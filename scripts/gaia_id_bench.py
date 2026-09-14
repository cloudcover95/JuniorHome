#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "gaia_proto.py").is_file():
        sys.path.insert(0, str(p))
        break
from ports.gaia_proto import handshake

notes = ["home dash", "buy oats", "cad title block", "flagstaff picnic"]
rows = []
for n in notes:
    t0 = time.perf_counter()
    env = handshake(n, job="dash-viewport")
    rows.append({
        "note": n,
        "ms": round((time.perf_counter() - t0) * 1000, 3),
        "schema_ok": env.get("schema_ok"),
        "verified": (env.get("identity") or {}).get("verified"),
        "issuer": (env.get("identity") or {}).get("issuer"),
        "hardcoded_operator": False,
        "area": (env.get("note") or {}).get("area"),
    })
print(json.dumps({"protocol": "goldend-osai-omega/1", "rows": rows}, indent=2))
