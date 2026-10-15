"""RPM + load-cell force → torque → power. Local only. No flash."""
from __future__ import annotations
import hashlib, json, math
from typing import Any
import numpy as np

RATE = 0.05

def torque_from_force(force_n: float, arm_m: float) -> float:
    return float(force_n) * float(arm_m)

def power_from_force(rpm: float, force_n: float, arm_m: float, corr: float = 1.0) -> dict[str, Any]:
    tq = torque_from_force(force_n, arm_m)
    pw = tq * float(rpm) * 2.0 * math.pi / 60.0 * float(corr)
    return {"rpm": rpm, "force_n": force_n, "arm_m": arm_m, "torque_nm": round(tq, 4),
            "power_w": round(pw, 4), "power_hp": round(pw / 745.7, 4), "corr": corr, "rate": RATE}

def inertia_alpha(ticks: list[float], i_kgm2: float) -> float:
    if len(ticks) < 3:
        return 0.0
    dt = np.diff(np.asarray(ticks, dtype=np.float64))
    if np.any(dt <= 0):
        return 0.0
    omega = 2.0 * math.pi / dt
    alpha = float(np.mean(np.diff(omega) / dt[:-1]))
    return float(i_kgm2) * alpha

def pack_run(row: dict[str, Any], prev: str = "") -> dict[str, Any]:
    text = json.dumps(row, sort_keys=True)
    body = hashlib.sha256(text.encode()).hexdigest()[:16]
    close = hashlib.sha256(f"{prev}|{body}|{text}".encode()).hexdigest()[:16]
    return {"row": row, "i2s": body, "close": close, "prev": prev, "rate": RATE}

if __name__ == "__main__":
    print(json.dumps(power_from_force(4500, 200.0, 0.25), indent=2))
