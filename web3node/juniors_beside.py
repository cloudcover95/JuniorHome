"""Block carries the trit pack and the signature beside it."""
import hashlib
from dilithium_py.ml_dsa import ML_DSA_65
def pack(data):
    xs = [(b - 128) / 128.0 for b in data[:32]]
    g = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / g)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)
def block(note, prev):
    pk, sk = ML_DSA_65.keygen()
    body = pack(note)
    sig = ML_DSA_65.sign(sk, body)
    ok = ML_DSA_65.verify(pk, body, sig)
    digest = hashlib.sha256(prev.encode() + body + sig).hexdigest()[:16]
    return {"prev": prev, "hash": digest, "trit": len(body), "sig": len(sig),
            "beside": True, "ok": ok}
def chain():
    a = block(b"cu-aurora", "0" * 16)
    b = block(b"cu-aurora|2", a["hash"])
    return {"blocks": [a, b], "linked": b["prev"] == a["hash"], "peers": 0, "network": False}
