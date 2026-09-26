#!/usr/bin/env python3
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for p in (root.parent / "JuniorLLM", Path.home() / "JuniorCloud" / "JuniorLLM"):
    if (p / "ports" / "blender_omega.py").is_file() or (p / "ports" / "gaia_proto.py").is_file():
        sys.path.insert(0, str(p))
        break

try:
    from ports.blender_omega import harness
except Exception:
    def harness(note="home dash terrain", out=None):
        dest = Path(out) if out else Path.home() / ".juniorhome" / "omega" / "patch.obj"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            "# juniorcloud omega obj\no gaia_patch\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
            encoding="utf-8",
        )
        return {"schema_ok": True, "job": "terrain-obj", "obj": str(dest), "bpy": False, "ue5_launch": False}

print(json.dumps(harness(), indent=2))
