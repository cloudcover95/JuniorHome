#!/usr/bin/env python3
"""Time OBJ write n=4 and n=16. No bpy."""
import json, sys, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for p in (root.parent / "JuniorLLM", Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "blender_omega.py").is_file():
        sys.path.insert(0, str(p))
        break

from ports.blender_omega import _obj, harness


def tick(n: int) -> dict:
    dest = Path.home() / ".juniorhome" / "omega" / f"bench_{n}.obj"
    t0 = time.perf_counter()
    _obj(dest, n=n)
    ms = (time.perf_counter() - t0) * 1000.0
    return {"n": n, "verts": n * n, "bytes": dest.stat().st_size, "ms": round(ms, 4)}


out = {"harness": harness(), "n4": tick(4), "n16": tick(16), "ue5_launch": False, "bpy": False}
print(json.dumps(out, indent=2))
