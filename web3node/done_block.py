"""A completed action becomes a trit block if the rails allow it."""
import hashlib
from done import done
from rails import allow
def block(kind, actor="dev"):
    row = done(kind)
    if not row["ok"]:
        return {"ok": False, "why": row["why"], "hash": None}
    gate = allow(actor, ["home"], 1)
    if not gate["ok"]:
        return {"ok": False, "why": gate["why"], "hash": None}
    body = f"{kind}|{actor}".encode()
    return {"ok": True, "kind": kind, "hash": hashlib.sha256(body).hexdigest()[:16],
            "trit": len(body), "ran": False, "network": False}
