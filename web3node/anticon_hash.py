"""AntiCon, then a trit block, then the hash failsafe."""
from anticon import fold
from failsafe import failsafe
from validate import make, valid
def guard(actor, hits):
    gate = fold(actor, ["node"], hits)
    if not gate["ok"]:
        return {"ok": False, "why": gate["why"], "backend": None}
    block = valid(make("|".join(hits)))
    if not block["ok"]:
        return {"ok": False, "why": "trit", "backend": None}
    back = failsafe(7)
    return {"ok": back["ok"], "why": "hash", "block": block["hash"], "tip": back["tip"],
            "backend": "hash", "lattice": False}
