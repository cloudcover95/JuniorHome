"""Validate a trit block. Tamper denies."""
import hashlib
def pack(text):
    raw = text.encode()[:32]
    xs = [(b - 128) / 128.0 for b in raw]
    g = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / g)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)
def make(text, prev="0" * 16):
    body = pack(text)
    return {"prev": prev, "hash": hashlib.sha256(prev.encode() + body).hexdigest()[:16], "body": body}
def valid(row):
    body = row["body"]
    if any(b not in (0, 1, 2) for b in body):
        return {"ok": False, "why": "not_trit"}
    digest = hashlib.sha256(row["prev"].encode() + body).hexdigest()[:16]
    if digest != row["hash"]:
        return {"ok": False, "why": "hash"}
    return {"ok": True, "hash": digest}
