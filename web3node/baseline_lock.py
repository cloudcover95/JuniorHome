"""Lock the baseline. Burst dies before a pack. Handshake is not in the close."""
import sys, time
from anticon import fold
from path import path
def lock():
    t0 = time.perf_counter()
    on = path("dev", ["gaia", "home"], True)
    burst = fold("dev", ["home"], ["gaia"] * 40)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (on["ok"] and on["rate"] == 0.05 and on["released"] == 0
          and on["mint"] is False and on["socket"] is False
          and burst["ok"] is False and burst["why"] == "burst")
    return {"ok": ok, "us": us, "peers": 0, "handshake_in_close": False, "temporal": False}
if __name__ == "__main__":
    import json
    row = lock()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
