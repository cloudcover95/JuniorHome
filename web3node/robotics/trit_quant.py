"""Trit quant. Numpy + stdlib. No JuniorCloud."""
from __future__ import annotations
import zlib
from typing import Any
import numpy as np

def _x(a):
    return np.asarray(a, dtype=np.float64).reshape(-1)

def absmean(a):
    x = _x(a)
    g = float(np.mean(np.abs(x))) or 1.0
    return np.clip(np.rint(x / g), -1, 1).astype(np.int8), g

def sign(a):
    x = _x(a)
    return np.where(x > 0, 1, np.where(x < 0, -1, 0)).astype(np.int8)

def threshold(a, t=None):
    x = _x(a)
    if t is None:
        t = float(np.std(x) * 0.25)
    return np.where(np.abs(x) < t, 0, np.sign(x)).astype(np.int8)

def pack(q):
    flat = np.clip(q.reshape(-1), -1, 1).astype(np.int8)
    codes = np.where(flat < 0, 0, np.where(flat == 0, 1, 2)).astype(np.uint8)
    pad = (-codes.size) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    packed = codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4) | (codes[3::4] << 6)
    return np.array([flat.size], dtype=np.uint32).tobytes() + packed.tobytes()

def bench(a):
    x = _x(a)
    qa, ga = absmean(x)
    qs, qt = sign(x), threshold(x)
    pa, ps, pt = pack(qa), pack(qs), pack(qt)
    return {"n": int(x.size), "float32": len(x.astype(np.float32).tobytes()),
            "absmean": {"gamma": ga, "sparsity": float((qa == 0).mean()), "bytes": len(pa), "zlib": len(zlib.compress(pa, 9))},
            "sign": {"sparsity": float((qs == 0).mean()), "bytes": len(ps), "zlib": len(zlib.compress(ps, 9))},
            "threshold": {"sparsity": float((qt == 0).mean()), "bytes": len(pt), "zlib": len(zlib.compress(pt, 9))}}
