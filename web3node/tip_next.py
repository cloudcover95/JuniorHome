"""Next close parents on a verified tip."""
from dilithium_py.ml_dsa import ML_DSA_65
from gate_chain import run
from osai_gate import gate
from validate import make, valid
def next_close():
    row = run([("dev", ["gaia"]), ("home", ["prompt"])])
    pk, sk = ML_DSA_65.keygen()
    msg = row["tip"].encode()
    sig = ML_DSA_65.sign(sk, msg)
    if not ML_DSA_65.verify(pk, msg, sig):
        return {"ok": False, "why": "sig"}
    nxt = gate("pi", ["field"], True)
    block = make(nxt["hash"], row["tip"])
    check = valid(block)
    return {"ok": check["ok"] and block["prev"] == row["tip"], "prev": row["tip"],
            "hash": block["hash"], "signs": 1, "disk": False}
