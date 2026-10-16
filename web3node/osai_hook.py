"""Home hook for OSai layer stack. Optional. No bench. No LUT."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def plan(m: int, k: int) -> dict[str, Any]:
    n = m * k
    if n < 512 * 512:
        layer = "t0_python"
    elif n < 1024 * 1024:
        layer = "t1_bitplane"
    else:
        layer = "t2_native"
    return {"m": m, "k": k, "layer": layer, "bench": False, "lut": False}

def write(m: int = 64, k: int = 64) -> Path:
    out = Path.home() / ".juniorhome" / "os" / "osai_plan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan(m, k), indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    print(write())
