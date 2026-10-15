"""Hardware abstraction stubs. Load cell, encoder, tach. Local only. No flash."""
from __future__ import annotations
import json
from typing import Any

def load_cell(raw: int, scale: float = 1.0, offset: float = 0.0) -> float:
    return (float(raw) - float(offset)) * float(scale)

def encoder(ticks: int, ppr: int = 1, dt_s: float = 1.0) -> float:
    if ppr < 1 or dt_s <= 0:
        return 0.0
    return (float(ticks) / float(ppr)) / float(dt_s) * 60.0

def tach(pulses: int, dt_s: float = 1.0) -> float:
    return encoder(pulses, ppr=1, dt_s=dt_s)

def sample(raw_force: int = 1000, ticks: int = 150, dt_s: float = 0.1) -> dict[str, Any]:
    return {"force_n": load_cell(raw_force, scale=0.2), "rpm": encoder(ticks, ppr=2, dt_s=dt_s),
            "write": False, "flash": False}

if __name__ == "__main__":
    print(json.dumps(sample(), indent=2))
