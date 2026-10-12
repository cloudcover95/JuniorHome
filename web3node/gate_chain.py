"""Gated closes, linked. A failed gate stops the chain."""
from osai_gate import gate
from validate import make, valid
def run(sessions):
    prev = "0" * 16
    n = 0
    for actor, hits in sessions:
        row = gate(actor, hits, True)
        if not row["ok"]:
            return {"ok": False, "why": row["why"], "n": n, "disk": False}
        block = make(row["hash"], prev)
        if not valid(block)["ok"]:
            return {"ok": False, "why": "trit", "n": n}
        prev = block["hash"]
        n += 1
    return {"ok": True, "n": n, "tip": prev, "disk": False, "ml_kem": False, "live": False}
