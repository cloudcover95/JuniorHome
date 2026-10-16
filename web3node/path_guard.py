"""Guard the path. Base rate held. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from path import path

def guard():
    t0 = time.perf_counter()
    good = path("dev", ["gaia", "home"], True)
    isolated = path("dev", ["gaia"], False)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good.get("rate") == 0.05
        and good.get("mint") is False
        and good.get("socket") is False
        and good.get("released") == 0
        and isolated.get("ok") is False
    )
    return {
        "ok": ok,
        "us": us,
        "rate": good.get("rate"),
        "juniors": good.get("juniors"),
        "tip": good.get("tip"),
        "released": 0,
        "mint": False,
        "socket": False,
        "held": True,
    }

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
