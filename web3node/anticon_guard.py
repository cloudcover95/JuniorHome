"""Guard the AntiCon hash path."""
import sys, time
from anticon_hash import guard
def check():
    t0 = time.perf_counter()
    ok = guard("dev", ["gaia", "home"])
    bot = guard("", ["gaia"])
    good = ok["ok"] and ok["backend"] == "hash" and bot["ok"] is False
    return {"ok": good, "us": round((time.perf_counter() - t0) * 1e6, 1)}
if __name__ == "__main__":
    import json
    row = check()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
