"""Custom BitNet/ternary kernel. Numpy is the array clock, not the kernel."""
from __future__ import annotations
from typing import Any
from bitnet_orig import absmean as absmean_list, i2s_pack
from tnn_layer import bitlinear
try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

def absmean_np(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    gamma = float(np.mean(np.abs(x))) or 1.0
    return np.clip(np.rint(x / gamma), -1, 1).astype(np.int8), gamma

def kernel_list(x, w):
    wq, gw = absmean_list(w)
    return {"path": "list", "layer": bitlinear(x, w), "pack_bytes": len(i2s_pack(wq)), "gamma": gw}

def kernel_np(x, w):
    q, g = absmean_np(w)
    return {"path": "numpy", "gamma": g, "sparsity": float((q == 0).mean()), "shape": list(q.shape)}

def pick_path():
    return "numpy" if HAVE_NP else "list"
