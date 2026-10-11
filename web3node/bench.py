"""Bench the local stack. This host is not an edge node."""
import hashlib, shutil, time
from dilithium_py.ml_dsa import ML_DSA_65
def pack(data):
    xs = [(b - 128) / 128.0 for b in data[:32]]
    g = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / g)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)
def bench():
    note = b"cu-aurora"
    t0 = time.perf_counter()
    body = pack(note)
    pack_us = (time.perf_counter() - t0) * 1e6
    t1 = time.perf_counter()
    digest = hashlib.sha256(body).hexdigest()[:16]
    hash_us = (time.perf_counter() - t1) * 1e6
    t2 = time.perf_counter()
    pk, sk = ML_DSA_65.keygen()
    sig = ML_DSA_65.sign(sk, body)
    ok = ML_DSA_65.verify(pk, body, sig)
    pq_us = (time.perf_counter() - t2) * 1e6
    return {"pack_us": round(pack_us, 1), "hash_us": round(hash_us, 1),
            "pq_us": round(pq_us, 1), "trit": len(body), "sig": len(sig), "ok": ok,
            "hash": digest, "iot": bool(shutil.which("nvidia-smi")), "network": False}
