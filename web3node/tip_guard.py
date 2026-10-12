"""Tip sign guard. Wrong key fails. One sign."""
from dilithium_py.ml_dsa import ML_DSA_65
from gate_chain import run
def guard():
    row = run([("dev", ["gaia"]), ("home", ["prompt"])])
    pk, sk = ML_DSA_65.keygen()
    other, _ = ML_DSA_65.keygen()
    msg = row["tip"].encode()
    sig = ML_DSA_65.sign(sk, msg)
    good = ML_DSA_65.verify(pk, msg, sig)
    bad = ML_DSA_65.verify(other, msg, sig)
    return {"ok": good and not bad and row["n"] == 2, "signs": 1, "disk": False}
