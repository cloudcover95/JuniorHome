"""Sign the close once. Not each hit."""
import time
from dilithium_py.ml_dsa import ML_DSA_65
from finish import finish
def sign(hits):
    row = finish(hits)
    if not row["ok"]:
        return {"ok": False, "why": "invalid"}
    t0 = time.perf_counter()
    pk, sk = ML_DSA_65.keygen()
    body = row["hash"].encode()
    sig = ML_DSA_65.sign(sk, body)
    ok = ML_DSA_65.verify(pk, body, sig)
    return {"ok": ok, "hits": row["hits"], "hash": row["hash"], "sig": len(sig),
            "signs": 1, "us": round((time.perf_counter() - t0) * 1e6, 1), "network": False}
