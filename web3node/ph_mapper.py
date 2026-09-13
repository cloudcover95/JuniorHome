"""Map PH apps to BitNet 1.58 / ternary kernel / T1 Spark profile."""
from __future__ import annotations
import json
from pathlib import Path
from compute_profiles import pick
from fieldcore_bridge import pick_juniorllm_port
from home_kernel import dispatch
from ph_apps import run_all
from ternary_kernel import kernel_list, pick_path

def map_app(name, payload, watts=45.0):
    task = f"{name} flagstaff homology bitnet"
    if name == "terrain":
        task, watts = "offline dem lidar omega", min(watts, 12)
    if name == "route" and watts >= 60:
        task = "ue5 spark world model"
    profile = pick(task, watts)
    kernel = dispatch(task, domain="terrain" if name == "terrain" else "agent", watts=watts)
    return {"app": name, "profile": profile, "surface": kernel.get("surface"),
            "ue_boot": kernel.get("ue_boot"), "port": pick_juniorllm_port("juniorosai field " + name),
            "quant": "ternary-1.58", "clock": pick_path(), "spark": profile == "T1_spark"}

def mapper(watts=45.0):
    apps = run_all()
    rows = [map_app(n, b, watts) for n, b in apps["apps"].items()]
    out = {"quant": "ternary-1.58", "clock": pick_path(), "watts": watts,
           "t1_requested": watts >= 60, "t1_granted": any(r["spark"] for r in rows),
           "bitlinear": kernel_list([0.2, -0.1, 0.3], [0.4, 0.0, -0.2]), "map": rows,
           "source": apps.get("source")}
    path = Path(__file__).resolve().parent / "vault" / "ph_mapper.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    out["out"] = str(path)
    return out

if __name__ == "__main__":
    t0, t1 = mapper(45), mapper(90)
    print(json.dumps({"T0": [r["profile"] for r in t0["map"]],
                      "T1_ask": [r["profile"] for r in t1["map"]]}, indent=2))
