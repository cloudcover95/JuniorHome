"""JuniorOS handheld: 10 m DEM + Omega capsule dry-run."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from dem_tile import ingest
from fieldcore_bridge import pick_juniorllm_port
from home_kernel import dispatch
from home_terraform import inject
from second_brain import write_note
from sparse_formats import compare_formats

def omega_dry_run(ply: Path, out_dir: Path) -> dict[str, Any]:
    try:
        import importlib
        return importlib.import_module("blender.jc_blender").run_job("omega-lidar", out_dir, trit=0, source=ply)
    except Exception:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = {"kind": "omega-lidar", "consumer": "JuniorOmega", "in": "ply", "out": "glb",
                "source": str(ply), "status": "staged",
                "note": "Blender / jc_blender not on path. PLY staged for handheld."}
        (out_dir / "job.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return spec

def feature(site="flagstaff"):
    tile = ingest(site)
    kernel = dispatch("offline dem slice lidar omega", domain="terrain", watts=12)
    port = pick_juniorllm_port("flagstaff field omega stonefield")
    tf = inject("junioros dem capsule handheld")
    ply = Path(tile["ply"])
    cap = omega_dry_run(ply, ply.parent / "capsule")
    note = write_note(Path(__file__).resolve().parent / "vault",
                      f"# JuniorOS DEM pulse\n\nsite: {tile['site']}\nz_mean: {tile['z_mean']}\nport: {port}\n")
    row = {"junioros": True, "handheld": True, "tile": tile,
           "kernel": {k: kernel.get(k) for k in ("profile", "surface", "ue_boot", "backend")},
           "port": port, "terraform": {k: tf.get(k) for k in ("port", "ok")},
           "omega": {k: cap.get(k) for k in ("kind", "status", "source", "note")},
           "obsidian": str(note)}
    out = Path(__file__).resolve().parent / "vault" / "junioros_feature.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(feature("flagstaff"), indent=2, default=str))
