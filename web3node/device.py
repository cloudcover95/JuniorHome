"""A device is a node. Same hit. No mine button."""
DEVICES = ("phone", "pi", "mac")
def device(kind, user):
    if kind not in DEVICES:
        return {"ok": False, "why": "unknown", "mine": False}
    return {"ok": True, "kind": kind, "user": user, "role": "node", "mine": False,
            "mint": False, "same_formula": True, "peers": 0, "network": False}
