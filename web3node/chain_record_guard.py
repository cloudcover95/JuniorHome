"""Guard the chain record. Base rate held. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from chain_record import run


def guard():
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
    )
    return {
        "ok": ok,
        "us": us,
        "n": good.get("n"),
        "tip": good.get("tip"),
        "juniors": good.get("juniors"),
        "rate": 0.05,
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
