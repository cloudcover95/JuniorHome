"""Bench the consolidated path. Rate, hold, and flags are locked."""
import sys, time
from path import path
def bench():
    t0 = time.perf_counter()
    on = path("dev", ["gaia", "home"], True)
    off = path("dev", ["gaia"], False)
    bot = path("", ["gaia"], True)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (on["ok"] and on["rate"] == 0.05 and on["released"] == 0
          and on["mint"] is False and on["socket"] is False
          and off["why"] == "isolated" and bot["why"] == "bot")
    return {"ok": ok, "us": us, "rate": on.get("rate"), "juniors": on.get("juniors")}
if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
