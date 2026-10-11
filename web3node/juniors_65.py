"""ML-DSA-65 over a trit rewrap. Vault name only. Not a chain."""
import hashlib, time
from dilithium_py.ml_dsa import ML_DSA_65
def pack(data):
    xs = [(b - 128) / 128.0 for b in data[:48]]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = []
    for x in xs:
        q = round(x / gamma)
        trits.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in trits), round(gamma, 3)
def block(note):
    t0 = time.perf_counter()
    pk, sk = ML_DSA_65.keygen()
    body, gamma = pack(note)
    sig = ML_DSA_65.sign(sk, body)
    ok = ML_DSA_65.verify(pk, body, sig)
    return {"scheme": "ML-DSA-65", "ok": ok, "pk": len(pk), "sig": len(sig),
            "trit_n": len(body), "gamma": gamma,
            "block": hashlib.sha256(body + sig).hexdigest()[:16],
            "us": round((time.perf_counter() - t0) * 1e6, 1),
            "vault": "obsidian", "synced": False, "peers": 0, "chain": False}
