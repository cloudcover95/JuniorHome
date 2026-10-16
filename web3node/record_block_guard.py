"""Guard the record block. Reward inside the pack. Base rate 0.05 held, mint false, socket false. Opt-in path. No model. No UE."""
import sys
import time
from record_block import record
from base_rate import BASE


def guard():
    t0 = time.perf_counter()
    good = record("dev", ["gaia", "home"])
    tip = record("home", ["prompt"], good.get("hash", "0" * 16))
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good.get("rate") == BASE
        and good.get("in_pack") is True
        and good.get("mint") is False
        and tip["ok"]
        and tip.get("prev") == good.get("hash")
        and tip.get("rate") == BASE
        and tip.get("mint") is False
    )
    return {
        "ok": ok,
        "us": us,
        "hash": good.get("hash"),
        "tip": tip.get("hash"),
        "prev": tip.get("prev"),
        "rate": BASE,
        "juniors": good.get("juniors", 0) + tip.get("juniors", 0),
        "released": 0,
        "mint": False,
        "socket": False,
        "held": True,
        "in_pack": True,
    }


if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
