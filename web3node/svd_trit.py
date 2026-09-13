"""Modern SVD. Numpy clock. MLX when present."""
from __future__ import annotations
from typing import Any
import numpy as np
from bitnet_orig import absmean
from trit_cache import pack_trits

def compute_hardware_svd(tensor_state: Any):
    try:
        import mlx.core as mx
        x = tensor_state if type(tensor_state).__module__.startswith("mlx") else mx.array(np.asarray(tensor_state))
        return mx.linalg.svd(x, stream=mx.cpu)
    except Exception:
        return np.linalg.svd(np.asarray(tensor_state, dtype=np.float64), full_matrices=False)

def retain(S, energy=0.90):
    s = np.asarray(S, dtype=np.float64).reshape(-1)
    tot = float((s * s).sum()) or 1.0
    acc = 0.0
    for k, v in enumerate(s, start=1):
        acc += float(v * v)
        if acc / tot >= energy:
            return k
    return int(s.size)

def residual(tensor_state, energy=0.90):
    U, S, Vt = compute_hardware_svd(tensor_state)
    u, s, vt = np.asarray(U), np.asarray(S).reshape(-1), np.asarray(Vt)
    k = retain(s, energy)
    recon = (u[:, :k] * s[:k]) @ vt[:k]
    src = np.asarray(tensor_state, dtype=np.float64)
    r = src - recon
    q, g = absmean(r.reshape(-1).tolist())
    return {"backend": "mlx" if type(U).__module__.startswith("mlx") else "numpy", "k": k,
            "energy": energy, "gamma": g, "trit_bytes": len(pack_trits(np.clip(np.rint(np.asarray(q)), -1, 1))),
            "residual_fro": float(np.linalg.norm(r))}
