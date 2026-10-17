"""Bench the true path under rate lock. Base 0.05 held. Opt-in, AntiCon, one trit pack, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from path import path
from base_rate import BASE


def bench():
    t0 = time.perf_counter()
    good = path("dev", ["gaia", "home"], True)
    isolated = path("dev", ["gaia"], False)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good.get("rate") == BASE
        and good.get("released") == 0
        and good.get("mint") is False
        and good.get("socket") is False
        and isolated["ok"] is False
        and isolated.get("why") == "isolated"
    )
    return {
        "ok": ok,
        "us": us,
        "rate": good.get("rate"),
        "juniors": good.get("juniors"),
        "released": 0,
        "mint": False,
        "socket": False,
        "held": True,
        "base": BASE,
    }


if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
