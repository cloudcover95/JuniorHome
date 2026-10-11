"""Sign the tip. Signature sits beside the trit pack."""
from dilithium_py.ml_dsa import ML_DSA_65
from hit_chain import chain
def signed():
    row = chain()
    tip = row["blocks"][-1]
    pk, sk = ML_DSA_65.keygen()
    body = bytes.fromhex(tip["receipt"])
    sig = ML_DSA_65.sign(sk, body)
    ok = ML_DSA_65.verify(pk, body, sig)
    return {"linked": row["linked"], "tip": tip["hash"], "sig": len(sig),
            "beside": True, "ok": ok, "mint": False, "network": False}
