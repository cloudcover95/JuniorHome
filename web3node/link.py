"""Link the next trit pack to the path tip. Base rate held."""
import sys
from path import path
from validate import make, valid
def link(actor, hits, opted_in):
    row = path(actor, hits, opted_in)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why", "path"), "mint": False, "socket": False}
    next_block = make("|" .join(hits) + "|next", row["tip"])
    check = valid(next_block)
    return {"ok": check["ok"], "prev": row["tip"], "hash": check["hash"],
            "rate": row["rate"], "released": row["released"],
            "juniors": row["juniors"], "mint": False, "socket": False}
if __name__ == "__main__":
    import json
    row = link("dev", ["gaia", "home"], True)
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
