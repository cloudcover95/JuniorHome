"""Bench the close-and-link. Tip advances. Pool stays held."""
import sys, time
from trit_close_link import close_link

def bench():
    t0 = time.perf_counter()
    a = close_link("dev", ["gaia"])
    b = close_link("home", ["prompt"], a.get("hash", "0" * 16))
    c = close_link("pi", ["field"], b.get("hash", "0" * 16))
    bad = close_link("", ["gaia"])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        a["ok"] and b["ok"] and c["ok"]
        and b["prev"] == a["hash"] and c["prev"] == b["hash"]
        and a["rate"] == 0.05 and b["released"] == 0 and c["mint"] is False
        and c["socket"] is False and bad["ok"] is False
        and a["packed_once"] is True
    )
    return {"ok": ok, "us": us, "tip": c.get("hash"), "n": 3}

if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
