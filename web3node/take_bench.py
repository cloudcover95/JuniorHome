"""Bench the take path. A bot stays at zero."""
import sys, time
from take import take
def bench():
    t0 = time.perf_counter()
    ok = take("dev", ["home"], ["gaia", "home"], [0.5])
    bot = take("", ["home"], ["gaia"], [0.5])
    return {"ok": ok["ok"] and ok["yield"] == 0.2 and bot["yield"] == 0,
            "yield": ok["yield"], "bot": bot["yield"],
            "us": round((time.perf_counter() - t0) * 1e6, 1), "mint": False}
if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
