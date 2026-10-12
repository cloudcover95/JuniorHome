"""Clip to P95, then trit pack. No disk."""
from validate import make, valid
def clip(text):
    raw = text.encode()[:32]
    if not raw:
        return ""
    xs = sorted(abs(b - 128) for b in raw)
    p95 = xs[max(0, int(len(xs) * 0.95) - 1)]
    out = []
    for b in raw:
        d = b - 128
        d = p95 if d > p95 else (-p95 if d < -p95 else d)
        out.append(128 + d)
    return bytes(out).decode("latin1")
def packed(text, prev="0" * 16):
    block = make(clip(text), prev)
    check = valid(block)
    return {"ok": check["ok"], "hash": check.get("hash"), "winsor": True,
            "ml_kem": False, "disk": False}
