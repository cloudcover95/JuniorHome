"""Manager. Ties channels, dyno, road, ticks, dash. Local only. No flash."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .channels import pack_sample
from .dyno import power_from_force, pack_run
from .road import power_from_speed
from .ticks import pack_ticks
from .dash import write as write_dash
from .autotune import recommend

def run(sample: dict[str, Any] | None = None, mode: str = "load") -> dict[str, Any]:
    sample = sample or {"rpm": 4500, "afr": 14.7, "tps": 0.4, "map": 1.0, "ect": 90, "iat": 30, "vss": 0, "gear": 3, "analog": 0.1}
    ch = pack_sample(sample)
    if mode == "road":
        dy = power_from_speed([10, 12, 14], [1.0, 1.0], 1500)
    elif mode == "ticks":
        dy = pack_ticks([0.0, 0.01, 0.02, 0.03], ppr=2)
    else:
        dy = power_from_force(float(sample.get("rpm", 0)), 200.0, 0.25)
    rec = recommend(float(sample.get("afr", 14.7)), 14.7)
    mesh = Path.home() / ".juniorhome" / "gaia_mesh"
    mesh.mkdir(parents=True, exist_ok=True)
    line = {"kind": "tuner", "mode": mode, "channels": ch, "dyno": dy, "autotune": rec}
    (mesh / "tuner.jsonl").open("a", encoding="utf-8").write(json.dumps(line) + "\n")
    dash = write_dash({"rpm": sample.get("rpm"), "power_hp": dy.get("power_hp"), "afr": sample.get("afr")})
    return {"ok": True, "close": ch["close"], "dash": str(dash), "flash": False}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
