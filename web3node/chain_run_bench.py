"""Guard the multi-close run."""
import sys, time
from chain_run import run
def bench():
    t0 = time.perf_counter()
    good = run([("dev", ["gaia"]), ("home", ["prompt"]), ("pi", ["field"])])
    bad = run([("dev", ["gaia"]), ("", ["prompt"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (good["ok"] and good["n"] == 3 and good["released"] == 0
          and good["rate"] == 0.05 and good["mint"] is False
          and bad["ok"] is False and bad["why"] == "bot")
    return {"ok": ok, "us": us, "n": good.get("n"), "tip": good.get("tip")}
if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
