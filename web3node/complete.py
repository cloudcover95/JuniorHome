"""Complete when the session passed AntiCon and packed once."""
from anticon import fold
from finish import finish
def complete(actor, hits):
    names = [k for k, n in hits.items() for _ in range(n)]
    gate = fold(actor, ["home"], names)
    if not gate["ok"]:
        return {"ok": False, "why": gate["why"], "complete": False}
    row = finish(hits)
    return {"ok": row["ok"], "complete": row["ok"], "actor": actor, "hits": row["hits"],
            "hash": row["hash"], "packed_once": True, "gas": False, "pow": False,
            "network": False}
