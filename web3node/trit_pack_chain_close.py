"""Close the trit-pack chain. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_pack_chain import trit_pack_chain
from base_rate import BASE
from validate import make, valid


def close_chain(sessions=None):
    chain = trit_pack_chain(sessions)
    if not chain.get("ok"):
        return {"ok": False, "why": chain.get("why", "chain"), "n": chain.get("n", 0), "mint": False, "socket": False}
    tip = chain.get("tip")
    block = make("|close|pack|0.05", tip)
    check = valid(block)
    if not check["ok"]:
        return {"ok": False, "why": "close", "n": chain.get("n"), "mint": False, "socket": False}
    if chain.get("rate") != BASE or not chain.get("held") or chain.get("released") != 0:
        return {"ok": False, "why": "held", "n": chain.get("n"), "mint": False, "socket": False}
    if chain.get("mint") is not False or chain.get("socket") is not False or chain.get("poll_set") is not False:
        return {"ok": False, "why": "flags", "n": chain.get("n"), "mint": False, "socket": False}
    return {
        "ok": True,
        "n": chain.get("n"),
        "tip": block["hash"],
        "prev": tip,
        "juniors": chain.get("juniors"),
        "rate": BASE,
        "released": 0,
        "held": True,
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
        "closed": True,
    }


def bench():
    t0 = time.perf_counter()
    good = close_chain()
    bad = close_chain([("dev", ["gaia"]), ("", ["prompt"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good["n"] == 3
        and good["closed"] is True
        and good["rate"] == 0.05
        and good["held"] is True
        and good["released"] == 0
        and good["mint"] is False
        and good["socket"] is False
        and good["poll_set"] is False
        and bad["ok"] is False
        and bad.get("why") == "bot"
    )
    return {
        "ok": ok,
        "us": us,
        "n": good.get("n"),
        "tip": good.get("tip"),
        "prev": good.get("prev"),
        "juniors": good.get("juniors"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "mint": False,
        "socket": False,
        "closed": good.get("closed"),
    }


if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
