"""All three ML-DSA sets over one trit pack. Signature does not fit in the pack."""
from dilithium_py.ml_dsa import ML_DSA_44, ML_DSA_65, ML_DSA_87
SETS = (("44", ML_DSA_44), ("65", ML_DSA_65), ("87", ML_DSA_87))
def pack(data):
    xs = [(b - 128) / 128.0 for b in data[:32]]
    g = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / g)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)
def share(note=b"cu-aurora"):
    body = pack(note)
    rows = []
    for name, mod in SETS:
        pk, sk = mod.keygen()
        sig = mod.sign(sk, body)
        ok = mod.verify(pk, body, sig)
        rows.append({"set": name, "ok": ok, "pk": len(pk), "sig": len(sig),
                     "trit": len(body), "fits_in_trit": len(sig) <= len(body)})
    return {"rows": rows, "slh_dsa": False, "peers": 0, "network": False}
