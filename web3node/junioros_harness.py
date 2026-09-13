"""JuniorOS harness. T0/T4 only. No UE."""
from __future__ import annotations
import json
from pathlib import Path
from agent_stack import run as agent_run
from barebones import tick as floor_tick
from fleet import isolate
from home_kernel import dispatch, probe
from off_caps import all_off
from tnn_layer import bitlinear
from trit_cache import compare
import numpy as np

def _dem_cached():
    meta = Path(__file__).resolve().parent / "vault" / "dem" / "flagstaff.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else None

def run():
    host = probe()
    kernel = dispatch("junioros harness numpy linux", domain="agent", watts=45)
    rng = np.random.default_rng(1)
    layer = bitlinear(rng.normal(size=64).tolist(), rng.normal(size=64).tolist())
    pack = compare(np.clip(np.rint(rng.normal(size=(16, 16))), -1, 1))
    agents = agent_run("junioros harness offline capsule")
    row = {"os": "JuniorOS", "host": host,
           "match": host.get("system") == "Linux" and not host.get("mlx"),
           "kernel": {k: kernel.get(k) for k in ("profile", "surface", "ue_boot", "backend")},
           "inference": {"bitlinear_y": layer["y"], "trit_pack": pack},
           "agents": {"ok": agents.get("ok"), "port": (agents.get("terraform") or {}).get("port")},
           "dem_cache": _dem_cached(), "off": all_off(),
           "fleet": {"edge_12w": isolate({"workload": "robot csi"}, 12),
                      "home_45w": isolate({"workload": "home"}, 45),
                      "floor": floor_tick([0.2, -0.1, 0.3], [0.4, 0.0, -0.2], 3.0)}}
    out = Path(__file__).resolve().parent / "vault" / "junioros_harness.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
