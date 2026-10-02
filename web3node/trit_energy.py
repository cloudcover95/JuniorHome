"""Trit-mesh energy rating. Not joules. Not a wattmeter.
energy = 1 - zeros/n. fail < 0.40, pass to 0.85, dense above.
"""
from __future__ import annotations
import json
from pathlib import Path
from trit_mesh import integrity
OUT = Path.home() / ".juniorhome" / "os" / "trit_energy.json"
LO, HI = 0.40, 0.85
def rate(note="JuniorOSai"):
    row = integrity(note)
    e = float(row["energy"])
    band = "fail" if e < LO else ("dense" if e > HI else "pass")
    body = {"protocol": "goldend-osai-omega/1", "energy": e, "lo": LO, "hi": HI,
            "band": band, "ok": band != "fail", "n": row["n"], "zeros": row["zeros"],
            "joules": False, "measured": False, "svd_1024": False,
            "bind": "127.0.0.1", "model_pull": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
