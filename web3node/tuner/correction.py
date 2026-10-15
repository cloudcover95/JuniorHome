"""SAE J1349 power correction. Local only. Default 1.0."""
from __future__ import annotations
import json, math
from typing import Any

def sae_j1349(temp_c: float = 25.0, pressure_kpa: float = 99.0, humidity: float = 0.0) -> float:
    p_dry = float(pressure_kpa) - float(humidity) * 0.01 * 3.17
    t = float(temp_c) + 273.15
    if p_dry <= 0 or t <= 0:
        return 1.0
    cf = (99.0 / p_dry) * math.sqrt(t / 298.15)
    return round(max(0.8, min(1.2, cf)), 4)

def apply(power_w: float, temp_c: float = 25.0, pressure_kpa: float = 99.0, humidity: float = 0.0) -> dict[str, Any]:
    cf = sae_j1349(temp_c, pressure_kpa, humidity)
    return {"power_w": round(power_w, 4), "corr": cf, "corrected_w": round(power_w * cf, 4),
            "corrected_hp": round(power_w * cf / 745.7, 4)}

if __name__ == "__main__":
    print(json.dumps(apply(100000, 30, 95, 40), indent=2))
