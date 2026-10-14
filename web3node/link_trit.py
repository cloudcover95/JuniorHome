"""Link closes as trit packs. One pack, base rate, pool held."""
from path import path
from validate import make, valid

def link(sessions):
    prev = "0" * 16
    tips = []
    juniors = 0.0
    for actor, hits in sessions:
        row = path(actor, hits, True)
        if not row["ok"]:
            return {"ok": False, "why": row.get("why"), "n": len(tips)}
        body = f"{prev}|{row['hash']}|{row['rate']}|{row['juniors']}"
        block = make(body, prev)
        check = valid(block)
        if not check["ok"]:
            return {"ok": False, "why": "trit", "n": len(tips)}
        tips.append(block["hash"])
        prev = block["hash"]
        juniors += row["juniors"]
    return {
        "ok": True,
        "n": len(tips),
        "tip": prev,
        "juniors": round(juniors, 4),
        "released": 0,
        "rate": 0.05,
        "mint": False,
        "socket": False,
        "one_pack": True,
    }
