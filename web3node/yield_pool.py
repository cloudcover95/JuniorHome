"""Yield is a share of the accrued Cloud pool. No mint."""
from wallet import wallet
def pool(cuts):
    return round(sum(cuts), 4)
def yield_of(user, hits, cuts):
    accrued = pool(cuts)
    w = wallet(user, hits)
    ask = w["juniors"]
    if ask > accrued:
        return {"ok": False, "why": "over_pool", "mint": False}
    return {"ok": True, "address": w["address"], "yield": ask, "pool": accrued,
            "left": round(accrued - ask, 4), "mint": False, "network": False}
