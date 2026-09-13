"""Measured Home ternary vs named open frameworks vs unpublished silicon claims."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from bitnet_orig import absmean
from compute_profiles import PROFILES
from tnn_layer import bitlinear
from trit_cache import compare

def _us(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6

def main():
    rng = np.random.default_rng(3)
    vec = rng.normal(size=256).tolist()
    w = rng.normal(size=256).tolist()
    mat = rng.normal(size=(32, 32))
    row = {
        "measured_home": {
            "absmean_256_us": round(_us(lambda: absmean(vec), 80), 3),
            "bitlinear_256_us": round(_us(lambda: bitlinear(vec, w), 80), 3),
            "trit_pack_32x32": compare(np.clip(np.rint(mat / (np.mean(np.abs(mat)) + 1e-7)), -1, 1)),
        },
        "silicon_claims_not_measured": {
            "taalas_hc1": "claimed ~17k tok/s Llama3.1-8B",
            "rule": "do not put claimed tok/s in Home SLAs",
        },
        "profiles": {k: v["name"] for k, v in PROFILES.items()},
    }
    path = Path(__file__).resolve().parent / "vault" / "bench_silicon_open.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))
    return row

if __name__ == "__main__":
    main()
