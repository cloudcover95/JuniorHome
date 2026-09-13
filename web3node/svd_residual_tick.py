"""SVD retain -> residual -> TernaryTDAMesh.tick.

Keeps compute_hardware_svd as the geometry source of truth.
Residual (A - A_k) is the only thing the ternary mesh sees.
CPU intent logits stay decision-only. Knockback is unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from svd_retain import (
    hardware_or_host_svd,
    reconstruct,
    retain_rank,
    retained_energy,
)
from ternary_tda import TernaryTDAMesh


def residual_tick(
    matrix: Any,
    mesh: TernaryTDAMesh | None = None,
    energy: float = 0.95,
    k: int | None = None,
) -> dict[str, Any]:
    arr = np.asarray(matrix, dtype=np.float64)
    u, s, vt = hardware_or_host_svd(arr)
    rank = k if k is not None else retain_rank(s, energy)
    approx = reconstruct(u, s, vt, rank)
    residual = arr - approx
    mesh = mesh or TernaryTDAMesh()
    tick = mesh.tick(residual)
    tick.update(
        {
            "svd_backend": "mlx" if _mlx() else "numpy",
            "k": rank,
            "full_rank": int(s.size),
            "retained_energy": retained_energy(s, rank),
            "residual_frob": float(np.linalg.norm(residual)),
            "approx_frob": float(np.linalg.norm(approx)),
        }
    )
    return tick


def _mlx() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    a = rng.normal(size=(48, 32))
    mesh = TernaryTDAMesh(drift_threshold=0.12)
    one = residual_tick(a, mesh=mesh, energy=0.90)
    two = residual_tick(a + rng.normal(size=a.shape) * 0.25, mesh=mesh, energy=0.90)
    keys = ("k", "retained_energy", "residual_frob", "drift", "qmark_collapse")
    print("[residual1]", {k: one[k] for k in keys})
    print("[residual2]", {k: two[k] for k in keys})
    print("[cpu_intent]", two["cpu_intent"])
