"""Reward is inside the pack. A changed cut moves the tip."""
from path import path
from validate import make, valid
def record(actor, hits, prev="0" * 16):
    row = path(actor, hits, True)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why")}
    body = f"{row['hash']}|{row['rate']}|{row['juniors']}"
    block = make(body, prev)
    check = valid(block)
    return {"ok": check["ok"], "hash": block["hash"], "prev": prev,
            "rate": row["rate"], "juniors": row["juniors"], "in_pack": True, "mint": False}
