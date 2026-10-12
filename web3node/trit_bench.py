"""Bench the multi trit-pack chain. Opt-in, AntiCon, one pack per close, base 0.05 held, mint false, socket false."""
import sys
import time
from trit_multi import trit_multi
from base_rate import BASE

def trit_bench(sessions=None, rounds=1):
    if sessions is None:
        sessions = [("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])]
    t0 = time.perf_counter()
    last = None
    for _ in range(max(1, rounds)):
        last = trit_multi(sessions)
        if not last["ok"]:
            return {"ok": False, "why": last.get("why", "multi"), "mint": False, "socket": False}
    us = round((time.perf_counter() - t0) * 1e6, 1)
    held = last.get("held") is True and last.get("rate") == BASE and last.get("released") == 0
    good = (
        last["ok"]
        and last["n"] == len(sessions)
        and held
        and last["mint"] is False
        and last["socket"] is False
        and last.get("poll_set") is False
        and last.get("packed_once") is True
    )
    return {
        "ok": good,
        "us": us,
        "n": last["n"],
        "tip": last.get("tip"),
        "rate": BASE,
        "held": held,
        "released": 0,
        "juniors": last.get("juniors"),
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
        "rounds": rounds,
    }

def guard():
    good = trit_bench()
    bad = trit_bench([("dev", ["gaia"]), ("", ["prompt"])])
    ok = (
        good["ok"]
        and good["n"] == 3
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and good["poll_set"] is False
        and bad["ok"] is False
        and bad.get("why") == "bot"
    )
    return {"ok": ok, "us": good.get("us"), "n": good.get("n"), "tip": good.get("tip"),
            "juniors": good.get("juniors"), "rate": good.get("rate")}

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
