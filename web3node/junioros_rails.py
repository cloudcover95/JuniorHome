"""JuniorOS rails contract. ON runs. OFF refuses."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from agent_stack import run as agent_run
from fieldcore_bridge import pick_juniorllm_port
from home_kernel import dispatch, probe
from home_terraform import inject
from off_caps import all_off
from tnn_layer import bitlinear
from trit_cache import compare

VAULT = Path(__file__).resolve().parent / "vault"
ON = ("home-clock", "numpy-bitlinear", "trit-pack", "agent-graph", "omega-stage",
      "dem-cache", "live-book-cache", "fieldcore-intent", "terraform-language")
OFF = ("ue-boot", "mlx-path", "blender-render", "cuda-path", "epqs-refetch", "yahoo-refetch")
TREES = {"JuniorHome": "clock + rails", "JuniorLLM": "ports/Flagstaff/agents",
         "JuniorOmega": "omega-lidar stage", "AGI_SDK": "agi-capsule",
         "StocksNode": "live_book cache", "FrameForge2D": "T0 game"}

def on_rail() -> dict[str, Any]:
    kernel = dispatch("junioros rails", domain="agent", watts=45)
    rng = np.random.default_rng(1)
    layer = bitlinear(rng.normal(size=64).tolist(), rng.normal(size=64).tolist())
    pack = compare(np.clip(np.rint(rng.normal(size=(16, 16))), -1, 1))
    agents = agent_run("junioros rails on")
    tf = inject("junioros rails field")
    dem = VAULT / "dem" / "flagstaff.json"
    book = VAULT / "live_book.json"
    return {
        "home-clock": {"state": "on", "profile": kernel["profile"], "backend": kernel["backend"]},
        "numpy-bitlinear": {"state": "on", "y": layer["y"], "n": layer["n"]},
        "trit-pack": {"state": "on", **pack},
        "agent-graph": {"state": "on" if agents.get("ok") else "fail", "patterns": agents.get("patterns")},
        "omega-stage": {"state": "on", "note": "staged; render OFF"},
        "dem-cache": {"state": "on" if dem.is_file() else "missing"},
        "live-book-cache": {"state": "on" if book.is_file() else "missing"},
        "fieldcore-intent": {"state": "on", "port": pick_juniorllm_port("field rails")},
        "terraform-language": {"state": "on" if tf.get("ok") else "fail", "port": tf.get("port")},
        "host": probe(),
    }

def architecture() -> dict[str, Any]:
    row = {"contract": "JuniorOS rails", "on_names": list(ON), "off_names": list(OFF),
           "on": on_rail(), "off": all_off(), "trees": TREES, "ue_boot": False, "fake_gpu": False}
    VAULT.mkdir(parents=True, exist_ok=True)
    out = VAULT / "junioros_rails.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(architecture(), indent=2, default=str))
