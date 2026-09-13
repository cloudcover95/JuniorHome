"""Bench Gemini threshold-quant vs live AbsMean + VR + SVD retain."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from macro_solver import gemini_threshold, load_matrix, solve
from ternary_tda import bitnet_quantize

def _us(fn, n: int) -> float:
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1e6

def main() -> dict:
    x = load_matrix()
    q_a, g = bitnet_quantize(x)
    q_t = gemini_threshold(x)
    q_a = np.asarray(q_a)
    row = {
        "absmean_us": round(_us(lambda: bitnet_quantize(x), 80), 3),
        "threshold_us": round(_us(lambda: gemini_threshold(x), 80), 3),
        "gamma": float(g),
        "agree": float((q_a == q_t).mean()),
        "absmean_zero": float((q_a == 0).mean()),
        "threshold_zero": float((q_t == 0).mean()),
        "pipeline": solve(x),
    }
    path = Path(__file__).resolve().parent / "bench_gemini_macro.json"
    path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    print(json.dumps(row, indent=2, default=str))
    return row

if __name__ == "__main__":
    main()
