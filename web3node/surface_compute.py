"""Compute a spun surface. Stdlib AbsMean."""
from __future__ import annotations

import json
from pathlib import Path

from surface_spin import spin
from tritquant import ticket

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "surface_compute.jsonl"


def compute(kind: str = "web", task: str = "JuniorOS") -> dict:
    spun = spin(kind, task)
    if not spun.get("ok"):
        return spun
    quant = ticket([0.2, -0.4, 0.1, 0.0, 0.3], [0.4, 0.0, -0.2, 0.1, 0.05])
    body = {"kind": kind, "sha3": spun["sha3"], "pack5": quant["pack5"], "y": quant["y"], "model_pull": False, "bind": "127.0.0.1"}
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, "sha3": body["sha3"]}) + "\n")
    return body
