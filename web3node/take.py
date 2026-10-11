"""Yield only after AntiCon passes."""
from anticon import fold
from yield_pool import yield_of
def take(actor, boxes, hits, cuts):
    row = fold(actor, boxes, hits)
    if not row["ok"]:
        return {"ok": False, "why": row["why"], "yield": 0, "mint": False}
    got = yield_of(actor, len(hits), cuts)
    if not got["ok"]:
        return {"ok": False, "why": got["why"], "yield": 0, "mint": False}
    return {"ok": True, "yield": got["yield"], "pool": got["pool"], "mint": False}
