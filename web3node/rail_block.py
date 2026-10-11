"""Rails before a trit block. A burst does not get a hash."""
import hashlib
from rails import allow
def block(actor, boxes, n):
    row = allow(actor, boxes, n)
    if not row["ok"]:
        return {"ok": False, "why": row["why"], "hash": None, "network": False}
    body = f"{actor}|{n}".encode()
    return {"ok": True, "hash": hashlib.sha256(body).hexdigest()[:16], "trit": len(body),
            "mint": False, "network": False}
