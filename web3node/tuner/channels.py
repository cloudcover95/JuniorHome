"""Winsor → trit → I2_S on engine channels. Local only."""
from __future__ import annotations
import hashlib, json
from typing import Any
import numpy as np

CHANNELS = ("rpm", "afr", "tps", "map", "ect", "iat", "vss", "gear", "analog")
RATE = 0.05

def _winsor(x, p=0.95):
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return a
    lo, hi = np.quantile(a, 1 - p), np.quantile(a, p)
    return np.clip(a, lo, hi)

def _trit(x):
    a = _winsor(x)
    g = float(np.mean(np.abs(a))) or 1.0
    return np.clip(np.rint(a / g), -1, 1).astype(np.int8), g

def _pack(q):
    flat = np.clip(q.reshape(-1), -1, 1).astype(np.int8)
    codes = np.where(flat < 0, 0, np.where(flat == 0, 1, 2)).astype(np.uint8)
    pad = (-codes.size) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    packed = codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4) | (codes[3::4] << 6)
    return packed.tobytes().hex()

def pack_sample(sample: dict[str, Any], prev: str = "") -> dict[str, Any]:
    vals = [float(sample.get(c, 0.0)) for c in CHANNELS]
    q, g = _trit(vals)
    body = _pack(q)
    text = json.dumps({c: sample.get(c) for c in CHANNELS}, sort_keys=True)
    close = hashlib.sha256(f"{prev}|{body}|{text}".encode()).hexdigest()[:16]
    return {"channels": CHANNELS, "trit": q.tolist(), "gamma": g, "i2s": body,
            "rate": RATE, "close": close, "prev": prev}

if __name__ == "__main__":
    s = {"rpm": 4500, "afr": 14.7, "tps": 0.4, "map": 1.0, "ect": 90, "iat": 30, "vss": 0, "gear": 3, "analog": 0.1}
    print(json.dumps(pack_sample(s), indent=2))
