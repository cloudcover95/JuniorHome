#!/usr/bin/env python3
"""Run OSai goldens + trit + probes. Sibling JuniorLLM if present."""
from __future__ import annotations

import json
import sys
from pathlib import Path

for p in (Path("../JuniorLLM").resolve(), Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "osai_goldens.py").is_file():
        sys.path.insert(0, str(p))
        break

out: dict = {"download": False, "kernel_patch": False, "ue5_launch": False}
try:
    from ports.osai_goldens import run as goldens
    from ports.home_harness import harness
    from junior_bitnet.winsor import pack

    out["goldens"] = goldens()
    out["harness"] = harness("flagstaff jay")
    out["winsor"] = pack([0.01, -2.0, 0.4])
    out["mode"] = "sibling-JuniorLLM"
except Exception as e:
    out["mode"] = "home-fallback"
    out["error"] = type(e).__name__ + ": " + str(e)[:160]
    try:
        from web3node.trit_tick import pack, tick
        from web3node.os_route import route
        from web3node.probe_future import probe

        out["trit"] = pack([0.01, -2.0, 0.4])
        out["route"] = route("flagstaff jay")
        out["probe"] = probe()
        out["tick"] = tick("stellar jay picnic", Path("/tmp/juniorhome_vault"))
    except Exception as e2:
        out["fallback_error"] = type(e2).__name__ + ": " + str(e2)[:160]

print(json.dumps(out, indent=2))
