"""Lock the tip path."""
import sys, time
from tip_guard import guard
from tip_next import next_close
def bench():
    t0 = time.perf_counter()
    g = guard()
    n = next_close()
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = g["ok"] and n["ok"] and n["signs"] == 1 and n["disk"] is False
    return {"ok": ok, "us": us, "prev": n.get("prev"), "signs": 1}
if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
