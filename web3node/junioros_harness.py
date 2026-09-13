"""JuniorOS harness: Linux x86_64 numpy-only. No MLX, no CUDA, no Blender."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from agent_stack import run as agent_run
from home_kernel import dispatch, probe
from tnn_layer import bitlinear
from trit_cache import compare

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
    row = {
        "os": "JuniorOS",
        "host": host,
        "match": host.get("system") == "Linux" and not host.get("mlx") and not host.get("cuda_cli") and host.get("blender_cli") is None,
        "kernel": {k: kernel.get(k) for k in ("profile", "surface", "ue_boot", "backend")},
        "inference": {"bitlinear_y": layer["y"], "bitlinear_n": layer["n"], "trit_pack": pack},
        "agents": {"ok": agents.get("ok"), "patterns": agents.get("patterns"),
                    "port": (agents.get("terraform") or {}).get("port")},
        "dem_cache": _dem_cached(),
        "markets_cache": (Path(__file__).resolve().parent / "vault" / "live_book.json").is_file(),
        "caps": {"ue_boot": False, "mlx": False, "blender_render": False,
                  "omega_stage": True, "numpy_infer": True, "trit_pack": True},
    }
    out = Path(__file__).resolve().parent / "vault" / "junioros_harness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
