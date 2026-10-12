"""Guard the reward-in-pack chain."""
import sys, time
from chain_record import run
def bench():
    t0 = time.perf_counter()
    row = run([("dev", ["gaia"]), ("home", ["prompt"]), ("pi", ["field"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (row["ok"] and row["n"] == 3 and row["in_pack"] is True
          and row["released"] == 0 and row["mint"] is False and row["socket"] is False)
    return {"ok": ok, "us": us, "tip": row.get("tip"), "juniors": row.get("juniors")}
if __name__ == "__main__":
    import json
    out = bench()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] else 1)
