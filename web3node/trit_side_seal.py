"""Trit side seal: balanced-ternary pack/unpack + side_commit. Pure math, no I/O."""
from __future__ import annotations
import hashlib

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False

def pack(trits):
    """Pack balanced trits (-1/0/1) as base-3 digits (5 per byte, 3**5=243<256)."""
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for t in reversed(trits[i:i + 5]):
            if t not in (-1, 0, 1):
                raise ValueError("trit must be -1, 0 or 1")
            v = v * 3 + (t + 1)
        out.append(v)
    return bytes(out)

def unpack(data, n):
    trits = []
    for b in data:
        for _ in range(5):
            trits.append(b % 3 - 1)
            b //= 3
    return trits[:n]

def side_commit(prev: str, pk: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]

def rate_of(x: float) -> float:
    return x * RATE

if __name__ == "__main__":
    t = [1, 0, -1, -1, 1, 0, 0, 1]
    p = pack(t)
    assert unpack(p, len(t)) == t
    c = side_commit("0" * 16, p, "junior")
    assert len(c) == 16 and c == side_commit("0" * 16, p, "junior")
    assert not (MINT or SOCKET or LAYER_ONE) and abs(rate_of(1.0) - 0.05) < 1e-12
    print({"pack": p.hex(), "commit": c, "ok": True})
