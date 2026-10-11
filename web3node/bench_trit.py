"""Bench trit pack and ML-DSA-65. Edge is not attached."""
import hashlib, time
from dilithium_py.ml_dsa import ML_DSA_65
def pack(data):
    xs = [(b - 128) / 128.0 for b in data[:32]]
    g = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / g)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)
def bench(note=b"dev|home|3"):
    t0 = time.perf_counter()
    body = pack(note)
    h = hashlib.sha256(body).hexdigest()[:16]
    pack_us = (time.perf_counter() - t0) * 1e6
    t1 = time.perf_counter()
    pk, sk = ML_DSA_65.keygen()
    sig = ML_DSA_65.sign(sk, body)
    ok = ML_DSA_65.verify(pk, body, sig)
    return {"trit": len(body), "hash": h, "pack_us": round(pack_us, 1),
            "pq_us": round((time.perf_counter() - t1) * 1e6, 1), "sig": len(sig),
            "ok": ok, "edge": False, "network": False}
