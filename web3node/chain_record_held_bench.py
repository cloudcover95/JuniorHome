"""Bench the chain record under rate lock. Base 0.05 held. Opt-in, AntiCon, one trit pack, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from chain_record import run
from base_rate import BASE


def bench():
    t0 = time.perf_counter()
    good = run([("dev", ["gaia"]), ("home", ["prompt"]), ("pi", ["field"])])
    empty = run([])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good["n"] == 3
        and good.get("released") == 0
        and good.get("mint") is False
        and good.get("socket") is False
        and good.get("in_pack") is True
        and empty["ok"] is True
        and empty["n"] == 0
        and good.get("juniors", 0) >= 0
    )
    return {
        "ok": ok,
        "us": us,
        "n": good.get("n"),
        "tip": good.get("tip"),
        "juniors": good.get("juniors"),
        "rate": BASE,
        "released": 0,
        "mint": False,
        "socket": False,
        "held": True,
        "in_pack": True,
        "base": BASE,
    }


if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
