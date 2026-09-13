"""Rank-k SVD retention helper. Does not replace svd_metal.py."""
from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "tolist"):
        return np.array(value.tolist(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def hardware_or_host_svd(matrix: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefer live compute_hardware_svd. Fall back to numpy (this host has no mlx)."""
    arr = _to_numpy(matrix)
    try:
        import mlx.core as mx  # type: ignore
        from svd_metal import compute_hardware_svd

        u, s, vt = compute_hardware_svd(mx.array(arr))
        return _to_numpy(u), _to_numpy(s).reshape(-1), _to_numpy(vt)
    except Exception:
        u, s, vt = np.linalg.svd(arr, full_matrices=False)
        return u, s, vt


def retain_rank(s: np.ndarray, energy: float = 0.95) -> int:
    """Smallest k with retained energy >= target. SVD stays; k is chosen, not dropped."""
    power = np.square(s)
    total = float(power.sum())
    if total <= 0:
        return 1
    cume = np.cumsum(power) / total
    k = int(np.searchsorted(cume, energy) + 1)
    return max(1, min(k, s.size))


def reconstruct(u: np.ndarray, s: np.ndarray, vt: np.ndarray, k: int) -> np.ndarray:
    return (u[:, :k] * s[:k]) @ vt[:k, :]


def retained_energy(s: np.ndarray, k: int) -> float:
    power = np.square(s)
    total = float(power.sum())
    if total <= 0:
        return 1.0
    return float(power[:k].sum() / total)
