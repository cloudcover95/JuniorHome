"""Several closes. Each links the last hash. Pool stays held."""
from path import path
from validate import make, valid
def run(sessions):
    prev = "0" * 16
    blocks = []
    juniors = 0.0
    for actor, hits in sessions:
        row = path(actor, hits, True)
        if not row["ok"]:
            return {"ok": False, "why": row.get("why"), "n": len(blocks)}
        block = make(row["hash"], prev)
        check = valid(block)
        if not check["ok"]:
            return {"ok": False, "why": "trit", "n": len(blocks)}
        blocks.append(block["hash"])
        prev = block["hash"]
        juniors += row["juniors"]
    return {"ok": True, "n": len(blocks), "tip": prev, "juniors": round(juniors, 4),
            "released": 0, "rate": 0.05, "mint": False, "socket": False}
