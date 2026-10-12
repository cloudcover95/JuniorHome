"""Multi-close on the reward pack. Each link carries the cut."""
from record_block import record
def run(sessions):
    prev = "0" * 16
    n = 0
    juniors = 0.0
    for actor, hits in sessions:
        row = record(actor, hits, prev)
        if not row["ok"]:
            return {"ok": False, "why": row.get("why"), "n": n}
        prev = row["hash"]
        juniors += row["juniors"]
        n += 1
    return {"ok": True, "n": n, "tip": prev, "juniors": round(juniors, 4),
            "in_pack": True, "released": 0, "mint": False, "socket": False}
