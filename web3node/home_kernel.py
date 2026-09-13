"""Cross-platform Home kernel. Domain-agnostic clock."""
from __future__ import annotations
import platform, shutil
from typing import Any
from compute_profiles import PROFILES, pick as pick_profile

DOMAINS = ("game", "cad", "field", "markets", "terrain", "llm", "agent")
SURFACES = {
    "T0_home": ("home-clock", "mlx-or-numpy", "blender-cmd", "frameforge2d"),
    "T1_spark": ("ue5", "world-model-small", "blender", "cuda"),
    "T2_discrete": ("ue5", "lingbot-class", "desk-terraform"),
    "T3_asic": ("frozen-trit-ticket",),
    "T4_mobile": ("trit-pack", "fieldcore-intent", "dem-slice"),
}

def probe() -> dict[str, Any]:
    mlx = False
    try:
        import mlx.core as mx  # noqa: F401
        mlx = True
    except Exception:
        pass
    return {"system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "mlx": mlx, "numpy": True,
            "cuda_cli": shutil.which("nvidia-smi") is not None,
            "blender_cli": shutil.which("blender")}

def backend(info=None) -> str:
    info = info or probe()
    return "mlx" if info.get("mlx") else "numpy"

def blender_cmd(job="agi-capsule", out_dir="blender_out"):
    exe = shutil.which("blender") or "blender"
    return [exe, "--background", "--python-expr", f"print({job!r}+{out_dir!r})"]

def allow_ue(profile: str) -> bool:
    return profile in ("T1_spark", "T2_discrete")

def dispatch(task: str, domain: str = "agent", watts: float = 45.0) -> dict[str, Any]:
    info = probe()
    domain = domain if domain in DOMAINS else "agent"
    profile = pick_profile(task, watts)
    if domain == "game" and profile == "T0_home":
        surface = "frameforge2d"
    elif domain in ("cad", "terrain") and profile == "T0_home":
        surface = "blender-cmd"
    elif domain == "terrain" and watts < 20:
        profile, surface = "T4_mobile", "dem-slice"
    elif domain == "markets":
        surface = "home-clock"
    else:
        surface = SURFACES[profile][0]
    return {"domain": domain, "profile": profile, "profile_name": PROFILES[profile]["name"],
            "backend": backend(info), "surface": surface, "surfaces": SURFACES[profile],
            "ue_boot": allow_ue(profile), "host": info,
            "blender_cmd": blender_cmd() if "blender" in surface else None, "agnostic": True}

if __name__ == "__main__":
    import json
    print(json.dumps({"probe": probe(), "dispatch": [
        dispatch("frameforge2d loopback", "game"),
        dispatch("omega title block", "cad"),
        dispatch("offline dem slice", "terrain", watts=12),
        dispatch("ue5 world model spark", "game", watts=90),
        dispatch("q_mark field ticker", "markets"),
        dispatch("taalas trit ticket", "llm"),
    ]}, indent=2))
