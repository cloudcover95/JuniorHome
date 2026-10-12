"""Two reward packs. Second prev is the first hash."""
from record_block import record
def chain():
    a = record("dev", ["gaia", "home"])
    b = record("home", ["prompt"], a["hash"])
    return {"ok": a["ok"] and b["ok"] and b["prev"] == a["hash"],
            "first": a["hash"], "second": b["hash"],
            "juniors": round(a["juniors"] + b["juniors"], 4),
            "in_pack": True, "mint": False}
