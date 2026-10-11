"""Chain guard. A wrong prev is not a link."""
import sys, time
from envelope_chain import chain, one
def guard():
    t0 = time.perf_counter()
    good = chain()
    a = one("dev", {"prompt": 4}, "0" * 16)
    bad = one("home", {"gaia": 2}, "f" * 16)
    linked = bad["ok"] and bad["prev"] == a["hash"]
    return {"ok": good["ok"] and not linked, "us": round((time.perf_counter() - t0) * 1e6, 1)}
if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
