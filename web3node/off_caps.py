"""OFF caps: refuse or stage. Do not fake GPU."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from home_kernel import allow_ue, blender_cmd, probe

def ue_boot(profile="T0_home"):
    if allow_ue(profile):
        return {"cap": "ue_boot", "state": "allowed", "ran": False, "note": "T1/T2 only — harness still does not launch UE"}
    return {"cap": "ue_boot", "state": "off", "ran": False, "note": "No UE boot on T0/T4."}

def mlx_path():
    host = probe()
    if host.get("mlx"):
        return {"cap": "mlx_path", "state": "on", "backend": "mlx"}
    return {"cap": "mlx_path", "state": "off", "backend": "numpy", "note": "mlx.core missing"}

def blender_render(out_dir="vault/dem/capsule"):
    host = probe()
    cmd = blender_cmd("omega-lidar", str(out_dir))
    if host.get("blender_cli"):
        return {"cap": "blender_render", "state": "available", "ran": False, "cmd": cmd}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    spec = {"cap": "blender_render", "state": "off", "ran": False, "cmd": cmd, "note": "no blender binary"}
    staged = Path(out_dir) / "OFF_blender.json"
    staged.write_text(__import__("json").dumps(spec, indent=2), encoding="utf-8")
    spec["staged"] = str(staged)
    return spec

def cuda_path():
    host = probe()
    if host.get("cuda_cli"):
        return {"cap": "cuda_path", "state": "on"}
    return {"cap": "cuda_path", "state": "off", "note": "no nvidia-smi — no fake GPU"}

def refetch(kind, force=False):
    vault = Path(__file__).resolve().parent / "vault"
    cached = {"epqs": (vault / "dem" / "flagstaff.json").is_file(), "yahoo": (vault / "live_book.json").is_file()}
    kind = (kind or "").lower()
    if kind not in cached:
        return {"cap": "refetch", "kind": kind, "state": "unknown", "ran": False}
    if not force:
        return {"cap": "refetch", "kind": kind, "state": "off", "ran": False, "cached": cached[kind]}
    return {"cap": "refetch", "kind": kind, "state": "forced", "ran": False}

def all_off():
    return {"ue_boot": ue_boot(), "mlx_path": mlx_path(), "blender_render": blender_render(),
            "cuda_path": cuda_path(), "epqs_refetch": refetch("epqs"), "yahoo_refetch": refetch("yahoo")}

if __name__ == "__main__":
    import json
    print(json.dumps(all_off(), indent=2))
