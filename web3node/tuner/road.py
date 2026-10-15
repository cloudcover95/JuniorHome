"""Road dyno from GPS speed delta. No hardware. Local only."""
from __future__ import annotations
import json, math
from typing import Any
import numpy as np

def power_from_speed(speeds_mps: list[float], dts: list[float], mass_kg: float, corr: float = 1.0) -> dict[str, Any]:
    if len(speeds_mps) < 2 or len(dts) < 1:
        return {"power_w": 0.0, "reason": "short"}
    v = np.asarray(speeds_mps, dtype=np.float64)
    dt = np.asarray(dts, dtype=np.float64)
    if np.any(dt <= 0):
        return {"power_w": 0.0, "reason": "bad_dt"}
    a = np.diff(v) / dt[:len(v)-1]
    f = float(mass_kg) * float(np.mean(a))
    p = f * float(np.mean(v)) * float(corr)
    return {"power_w": round(p, 4), "power_hp": round(p / 745.7, 4), "accel": round(float(np.mean(a)), 4),
            "mass_kg": mass_kg, "corr": corr, "mode": "road"}

if __name__ == "__main__":
    print(json.dumps(power_from_speed([10, 12, 14], [1.0, 1.0], 1500), indent=2))
