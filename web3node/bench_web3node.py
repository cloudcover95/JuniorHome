#!/usr/bin/env python3
"""Bench SVD retention, ternary residual, VR 1-skeleton, trit cache."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from svd_residual_tick import residual_tick
from svd_retain import hardware_or_host_svd, retain_rank, retained_energy
from ternary_tda import TernaryTDAMesh
from trit_cache import compare
from vietoris_rips import vr_summary


def _time(fn, loops: int = 5) -> float:
    start = time.perf_counter()
    for _ in range(loops):
        fn()
    return (time.perf_counter() - start) / loops


def bench(dim: int = 48) -> dict:
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(dim, dim)).astype(np.float64)
    mesh = TernaryTDAMesh()

    def svd_only():
        hardware_or_host_svd(matrix)

    def residual():
        residual_tick(matrix, mesh=mesh, energy=0.95)

    points = matrix[:24, :8]
    row = {
        "dim": dim,
        "svd_s": _time(svd_only),
        "residual_tick_s": _time(residual, loops=3),
        "vr": vr_summary(points),
        "trit": compare(np.clip(np.rint(matrix / (np.mean(np.abs(matrix)) + 1e-7)), -1, 1)),
    }
    u, s, vt = hardware_or_host_svd(matrix)
    k = retain_rank(s, 0.95)
    row["retain"] = {"k": k, "energy": retained_energy(s, k), "full": int(s.size)}
    return row


if __name__ == "__main__":
    result = bench()
    out = Path(__file__).resolve().parent / "bench_last.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("[wrote]", out)
