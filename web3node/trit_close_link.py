"""Close once, pack the cut, link the tip. Pool held."""
from path import path
from validate import make, valid

def close_link(actor, hits, prev="0" * 16):
    row = path(actor, hits, True)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why")}
    body = f"{row['hash']}|{row['rate']}|{row['juniors']}|held"
    block = make(body, prev)
    check = valid(block)
    if not check["ok"]:
        return {"ok": False, "why": "trit"}
    return {
        "ok": True,
        "hash": block["hash"],
        "prev": prev,
        "rate": row["rate"],
        "juniors": row["juniors"],
        "released": 0,
        "mint": False,
        "socket": False,
        "packed_once": True,
        "in_pack": True,
    }
