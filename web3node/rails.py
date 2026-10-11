"""Hit rails. Not a firewall."""
CAP = 32
def allow(actor, boxes, n):
    if n > CAP:
        return {"ok": False, "why": "burst", "firewall": False}
    if len(boxes) != len(set(boxes)):
        return {"ok": False, "why": "multi_box", "firewall": False}
    if not actor:
        return {"ok": False, "why": "bot", "firewall": False}
    return {"ok": True, "actor": actor, "n": n, "cap": CAP, "firewall": False}
