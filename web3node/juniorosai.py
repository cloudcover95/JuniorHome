"""JuniorOSai Home bench + card."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from prototype_engine import engine
from ternary_kernel import kernel_list, kernel_np

CARD = {
    "name": "JuniorOSai", "kind": "bitnet-native", "quant": "ternary-1.58",
    "backend": "numpy-or-mlx", "max_download_gb": 0.0,
    "params_live": "kernel-only",
    "params_ship_target": {
        "T0": "0B download — BitLinear + trit ticket",
        "T0+gguf": "BitNet-2B4T I2_S ~1.5 GB if present",
        "T1_spark": "Q4 4B-8B, 24-128 GB unified",
    },
    "context": {"flagstaff_total_chars": 2176},
}

def bench():
    rng = np.random.default_rng(9)
    vec = rng.normal(size=256).tolist(); w = rng.normal(size=256).tolist()
    mat = rng.normal(size=(64, 64))
    t0 = time.perf_counter()
    for _ in range(30):
        kernel_list(vec, w)
    list_us = (time.perf_counter() - t0) / 30 * 1e6
    t0 = time.perf_counter()
    for _ in range(30):
        kernel_np(mat, mat)
    np_us = (time.perf_counter() - t0) / 30 * 1e6
    row = {"card": CARD, "engine": engine("juniorosai field"),
           "bench": {"list_kernel_256_us": round(list_us, 3),
                      "numpy_absmean_64x64_us": round(np_us, 3),
                      "trit_ticket_kib_64x64": 1.0},
           "ship_gate": {"weights_on_disk": False, "kernel_ready": True, "card_ready": True,
                          "production": "prototype — kernel+card, not a 2B checkpoint"}}
    out = Path(__file__).resolve().parent / "vault" / "juniorosai.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(bench(), indent=2, default=str))
