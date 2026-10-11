"""AntiCon. A fold over node hits. Not a network filter."""
from rails import allow
def fold(actor, boxes, hits):
    row = allow(actor, boxes, len(hits))
    if not row["ok"]:
        return {"ok": False, "why": row["why"], "fold": "anticon", "yield": False}
    return {"ok": True, "fold": "anticon", "n": len(hits), "actor": actor,
            "yield": True, "firewall": False}
