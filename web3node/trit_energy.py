"""Trit-mesh energy rating. Disk only if the envelope is closed."""
from __future__ import annotations
import json
from pathlib import Path
from trit_mesh import integrity
from write_gate import commit
OUT = Path.home() / ".juniorhome" / "os" / "trit_energy.json"
LO, HI = 0.40, 0.85
def rate(note="JuniorOSai", env="open"):
    row = integrity(note)
    e = float(row["energy"])
    band = "fail" if e < LO else ("dense" if e > HI else "pass")
    body = {"protocol": "goldend-osai-omega/1", "energy": e, "lo": LO, "hi": HI,
            "band": band, "ok": band != "fail", "n": row["n"], "zeros": row["zeros"],
            "joules": False, "measured": False, "svd_1024": False,
            "bind": "127.0.0.1", "model_pull": False}
    written = commit(env, OUT, json.dumps(body) + "\n")
    body["disk"] = written["disk"]
    body["why"] = written["why"]
    return body
