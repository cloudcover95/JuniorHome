"""Receipt becomes a trit block. Parent is local. No peers."""
import hashlib
from hit import hit
def pack(data):
    xs = [(b - 128) / 128.0 for b in data[:32]]
    g = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / g)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)
def block(content, nodes, prev="0" * 16):
    row = hit(content, nodes)
    if not row["ok"]:
        return {"ok": False, "why": row["why"], "network": False}
    body = pack(bytes.fromhex(row["receipt"]))
    digest = hashlib.sha256(prev.encode() + body).hexdigest()[:16]
    return {"ok": True, "prev": prev, "hash": digest, "trit": len(body),
            "receipt": row["receipt"], "cloud": row["cloud"], "mint": False, "network": False}
