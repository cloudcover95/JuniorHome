"""Edge bench: original-Python AbsMean vs numpy ternary + Flagstaff tick + terraform inject."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from bitnet_orig import absmean, i2s_pack, sign
from fieldcore_bridge import run_flagstaff
from home_terraform import inject
from ternary_tda import bitnet_quantize
from trit_cache import compare


def _us(fn, n: int) -> float:
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1e6


def main() -> dict:
    rng = np.random.default_rng(21)
    vec = rng.normal(size=240).astype(np.float64)
    xs = vec.tolist()
    mat = rng.normal(size=(64, 64)).astype(np.float64)
    orig_t, orig_s = absmean(xs)
    _np_q, np_g = bitnet_quantize(vec.reshape(16, 15))
    row = {
        "orig_absmean_us": round(_us(lambda: absmean(xs), 200), 3),
        "orig_sign_us": round(_us(lambda: sign(xs), 200), 3),
        "orig_i2s_us": round(_us(lambda: i2s_pack(orig_t), 200), 3),
        "numpy_quant_us": round(_us(lambda: bitnet_quantize(mat), 80), 3),
        "orig_scale": orig_s,
        "orig_sparsity": orig_t.count(0) / len(orig_t),
        "orig_packed": len(i2s_pack(orig_t)),
        "numpy_gamma": np_g,
        "trit": compare(np.clip(np.rint(mat / (np.mean(np.abs(mat)) + 1e-7)), -1, 1)),
        "flagstaff": run_flagstaff("flagstaff dry V4 crimp"),
        "terraform": inject("flagstaff dry V4 crimp home local llm"),
    }
    out = Path(__file__).resolve().parent / "bench_bitnet_edge.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: row[k] for k in row if k not in ("flagstaff",)}, indent=2, default=str))
    print("[flagstaff port]", row["flagstaff"]["juniorllm_port"])
    print("[wrote]", out)
    return row


if __name__ == "__main__":
    main()
