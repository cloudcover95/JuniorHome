"""Trit pack link. One close, one pack, base rate, pool held, then link. Guard denies a broken prev."""
import sys
from path import path
from validate import make, valid

def link(actor, hits, prev="0" * 16, opted_in=True):
    row = path(actor, hits, opted_in)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why"), "linked": False}
    block = make(row["hash"], prev)
    check = valid(block)
    if not check["ok"]:
        return {"ok": False, "why": "link", "linked": False}
    return {"ok": True, "hash": block["hash"], "prev": prev, "rate": row["rate"],
            "released": row["released"], "mint": False, "socket": False, "linked": True}

def guard():
    a = link("dev", ["gaia"])
    b = link("home", ["deck"], a["hash"])
    tampered = {"prev": "0" * 16, "hash": b["hash"], "body": make("x")["body"]}
    bad = valid(tampered)
    return {"ok": a["ok"] and b["ok"] and b["prev"] == a["hash"] and not bad["ok"],
            "tip": b["hash"], "rate": 0.05, "held": a["released"] == 0,
            "mint": False, "socket": False}

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
