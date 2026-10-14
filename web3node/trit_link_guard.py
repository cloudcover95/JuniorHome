"""Link two trit closes. Pool held. Rate base. Mint and socket stay false."""
import sys, time
from path import path
from validate import make, valid
def link():
    t0 = time.perf_counter()
    a = path("dev", ["gaia"], True)
    if not a.get("ok"):
        return {"ok": False, "why": a.get("why")}
    b = path("home", ["prompt"], True)
    if not b.get("ok"):
        return {"ok": False, "why": b.get("why")}
    block = make(b["hash"], a["hash"])
    check = valid(block)
    ok = (a["ok"] and b["ok"] and check["ok"]
          and a["rate"] == 0.05 and b["rate"] == 0.05
          and a["released"] == 0 and b["released"] == 0
          and a["mint"] is False and b["mint"] is False
          and a["socket"] is False and b["socket"] is False)
    return {"ok": ok, "us": round((time.perf_counter() - t0) * 1e6, 1),
            "tip": block["hash"], "rate": 0.05, "released": 0,
            "mint": False, "socket": False}
if __name__ == "__main__":
    import json
    row = link()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
