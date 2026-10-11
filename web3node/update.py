"""An update is a finished session. Not one action."""
from finish import finish
def update(user, hits):
    row = finish(hits)
    if not row["ok"] or row["hits"] < 1:
        return {"ok": False, "why": "no_session", "protocol_changed": False}
    return {"ok": True, "user": user, "hits": row["hits"], "hash": row["hash"],
            "packed_once": True, "pow": False, "protocol_changed": False, "network": False}
