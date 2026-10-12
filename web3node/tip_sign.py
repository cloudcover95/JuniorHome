"""One ML-DSA sign on the tip. Not one per close."""
import time
from dilithium_py.ml_dsa import ML_DSA_65
from gate_chain import run
def sign_tip():
    row = run([("dev", ["gaia"]), ("home", ["prompt"])])
    if not row["ok"]:
        return {"ok": False, "why": "chain"}
    pk, sk = ML_DSA_65.keygen()
    msg = row["tip"].encode()
    t0 = time.perf_counter()
    sig = ML_DSA_65.sign(sk, msg)
    ok = ML_DSA_65.verify(pk, msg, sig)
    return {"ok": ok, "closes": row["n"], "signs": 1, "tip": row["tip"],
            "ms": round((time.perf_counter() - t0) * 1000, 1), "disk": False}
