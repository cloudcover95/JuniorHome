"""Trit close on the true path. Opt-in, AntiCon, one trit pack, base 0.05 held, no mint."""
import sys
from close import close
from path import path
from validate import make, valid

def trit_close(actor, hits, opted_in):
    if not opted_in:
        return {"ok": False, "why": "isolated", "mint": False, "socket": False}
    row = path(actor, hits, opted_in)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why", "path"), "mint": False, "socket": False}
    packed = close({h: 1 for h in hits})
    block = make("|" .join(hits) + "|close", row["tip"])
    check = valid(block)
    held = row.get("released", 1) == 0
    good = (check["ok"] and packed["packed_once"] and
            row["rate"] == 0.05 and held and
            row["mint"] is False and row["socket"] is False)
    return {
        "ok": good,
        "rate": row["rate"],
        "held": held,
        "hash": check.get("hash"),
        "prev": row["tip"],
        "packed_once": True,
        "juniors": row.get("juniors"),
        "mint": False,
        "socket": False,
        "poll_set": False,
    }

if __name__ == "__main__":
    import json
    row = trit_close("dev", ["gaia", "home"], True)
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
