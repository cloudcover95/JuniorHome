"""Omega capsule dry-run. Stages job + PLY. No bpy."""
from __future__ import annotations
import json
from pathlib import Path
from home_kernel import blender_cmd, dispatch
OUT = Path(__file__).resolve().parent / "vault" / "omega_capsule"

def _grid_from_dem():
    tile = Path(__file__).resolve().parent / "vault" / "dem" / "flagstaff.ply"
    if tile.is_file():
        pts = []
        for line in tile.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                try:
                    pts.append(tuple(map(float, parts)))
                except ValueError:
                    pass
        if pts:
            return pts
    return [(i % 6 - 2.5, 0.0, i // 6 - 2.5) for i in range(36)]

def dry_run(kind="agi-capsule"):
    try:
        import importlib
        spec = importlib.import_module("blender.jc_blender").run_job(kind, OUT, trit=0)
        spec["host"] = "junioromega.jc_blender"
        return spec
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    pts = _grid_from_dem()
    lines = ["ply", "format ascii 1.0", f"element vertex {len(pts)}", "property float x", "property float y", "property float z", "end_header"]
    lines += [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in pts]
    ply = OUT / "lidar.ply"
    ply.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spec = {"kind": kind, "status": "staged", "host": "home-twin", "source": str(ply),
            "blender_cmd": blender_cmd(kind), "kernel": dispatch("omega capsule", "cad"),
            "consumers": ["JuniorOmega", "AGI_SDK", "JuniorDrive"]}
    (OUT / "job.json").write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")
    return spec

if __name__ == "__main__":
    print(json.dumps(dry_run(), indent=2))
