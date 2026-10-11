"""Name is bound to a key. Wrong key fails. No socket."""
from dilithium_py.ml_dsa import ML_DSA_65
def enroll(name):
    pk, sk = ML_DSA_65.keygen()
    return {"name": name, "pk": pk, "sk": sk}
def hello(row, parent):
    msg = f"{row['name']}|{parent}".encode()
    return {"name": row["name"], "parent": parent, "sig": ML_DSA_65.sign(row["sk"], msg), "msg": msg}
def accept(book, hel):
    pk = book.get(hel["name"])
    if pk is None:
        return {"ok": False, "why": "not_enrolled", "socket": False}
    ok = ML_DSA_65.verify(pk, hel["msg"], hel["sig"])
    return {"ok": ok, "why": "ok" if ok else "bad_sig", "socket": False}
