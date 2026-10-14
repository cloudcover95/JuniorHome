"""Bench the trit link. Opt-in, AntiCon, one pack, base 0.05, pool held."""
import sys, time
from link_trit import link

def bench():
    t0 = time.perf_counter()
    good = link([("dev", ["gaia"]), ("home", ["prompt"]), ("pi", ["field"])])
    bad = link([("dev", ["gaia"]), ("", ["prompt"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good["n"] == 3
        and good["released"] == 0
        and good["rate"] == 0.05
        and good["mint"] is False
        and good["socket"] is False
        and good.get("one_pack") is True
        and bad["ok"] is False
        and bad["why"] == "bot"
    )
    return {"ok": ok, "us": us, "n": good.get("n"), "tip": good.get("tip"), "juniors": good.get("juniors")}

if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
