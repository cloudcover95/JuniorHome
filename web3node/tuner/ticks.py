"""Timestamped encoder ticks → RPM and alpha. Local only."""
from __future__ import annotations
import json
from typing import Any
import numpy as np

def rpm_from_ticks(ticks: list[float], pulses_per_rev: int = 1) -> float:
    if len(ticks) < 2 or pulses_per_rev < 1:
        return 0.0
    dt = np.diff(np.asarray(ticks, dtype=np.float64))
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    period = float(np.mean(dt)) * pulses_per_rev
    return 60.0 / period if period > 0 else 0.0

def alpha_from_ticks(ticks: list[float]) -> float:
    if len(ticks) < 3:
        return 0.0
    dt = np.diff(np.asarray(ticks, dtype=np.float64))
    if np.any(dt <= 0):
        return 0.0
    omega = 2.0 * np.pi / dt
    return float(np.mean(np.diff(omega) / dt[:-1]))

def pack_ticks(ticks: list[float], ppr: int = 1) -> dict[str, Any]:
    return {"rpm": round(rpm_from_ticks(ticks, ppr), 3), "alpha": round(alpha_from_ticks(ticks), 6), "n": len(ticks)}

if __name__ == "__main__":
    print(json.dumps(pack_ticks([0.0, 0.01, 0.02, 0.03], ppr=2), indent=2))
