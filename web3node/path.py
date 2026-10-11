"""The path that is true. Counter, one pack, base rate, hash prove."""
import sys
from anticon import fold
from failsafe import failsafe
from fund import fund
from reward import reward
from validate import make, valid
def path(actor, hits, opted_in):
    if not opted_in:
        return {"ok": False, "why": "isolated"}
    gate = fold(actor, ["node"], hits)
    if not gate["ok"]:
        return {"ok": False, "why": gate["why"]}
    block = valid(make("|".join(hits)))
    if not block["ok"]:
        return {"ok": False, "why": "trit"}
    back = failsafe(7)
    pay = reward(len(hits), True)
    pool = fund(pay["juniors"], False)
    return {"ok": back["ok"] and pay["ok"], "hash": block["hash"], "tip": back["tip"],
            "rate": pool["rate"], "juniors": pay["juniors"], "released": pool["released"],
            "mint": False, "socket": False}
if __name__ == "__main__":
    import json
    row = path("dev", ["gaia", "home"], True)
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
