"""ML-DSA-44 on a trit pack. Needs dilithium-py. Not a chain."""
import hashlib
from dilithium_py.ml_dsa import ML_DSA_44
def pack(text):
    xs = [((ord(c) % 5) - 2) / 2.0 for c in (text or "x")[:32]]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)
def sign(text):
    pk, sk = ML_DSA_44.keygen()
    body = pack(text)
    sig = ML_DSA_44.sign(sk, body)
    ok = ML_DSA_44.verify(pk, body, sig)
    return {"scheme": "ML-DSA-44", "ok": ok, "pk_len": len(pk), "sig_len": len(sig),
            "n": len(body), "pq_ready": ok, "block": hashlib.sha256(body + sig).hexdigest()[:16],
            "dormant": not ok, "sweep": False, "chain": False}
