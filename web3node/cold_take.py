"""Cold denies a take. Presenting opens it."""
from take import take
def take_cold(actor, hits, cuts, cold):
    if cold:
        return {"ok": False, "why": "cold", "yield": 0, "key_exported": False}
    return take(actor, ["home"], hits, cuts)
