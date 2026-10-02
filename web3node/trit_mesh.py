"""Mesh integrity as trit energy. Does not write. The write gate does."""
from __future__ import annotations
def integrity(note="JuniorHome"):
    xs = [((ord(c) % 5) - 2) / 2.0 for c in note[:32]] or [0.0]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = []
    for x in xs:
        q = round(x / gamma)
        trits.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    zeros = trits.count(0)
    return {"protocol": "goldend-osai-omega/1", "n": len(trits), "zeros": zeros,
            "energy": round(1.0 - zeros / len(trits), 3), "svd_1024": False,
            "bind": "127.0.0.1", "model_pull": False}
