"""Tuner hook into the layer stack. Optional. No bench. No LUT."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def plan(m: int, k: int) -> str:
    n = m * k
    if n < 512 * 512:
        return "t0_python"
    if n < 1024 * 1024:
        return "t1_bitplane"
    return "t2_native"

def write(m: int = 64, k: int = 64) -> Path:
    out = Path.home() / ".juniorhome" / "os" / "tuner_stack.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"m": m, "k": k, "layer": plan(m, k), "bench": False, "lut": False}, indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    print(write())
