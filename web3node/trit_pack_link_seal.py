"""Seal a linked trit-pack close. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_pack_link_guard import link_close
from validate import make, valid
from base_rate import BASE


def seal(sessions):
    linked = link_close(sessions)
    if not linked.get("ok"):
        return {"ok": False, "why": linked.get("why", "link"), "sealed": False}
    tip = linked.get("tip")
    block = make("seal|" + tip, tip)
    check = valid(block)
    if not check["ok"]:
        return {"ok": False, "why": "seal", "sealed": False}
    if linked.get("rate") != BASE or linked.get("released") != 0 or not linked.get("held"):
        return {"ok": False, "why": "held", "sealed": False}
    if linked.get("mint") is not False or linked.get("socket") is not False:
        return {"ok": False, "why": "flags", "sealed": False}
    return {
        "ok": True,
        "n": linked.get("n"),
        "tip": check["hash"],
        "link_tip": tip,
        "juniors": linked.get("juniors"),
        "rate": BASE,
        "released": 0,
        "held": True,
        "mint": False,
        "socket": False,
        "poll_set": False,
        "sealed": True,
    }


def bench():
    t0 = time.perf_counter()
    good = seal([("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])])
    empty = seal([])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good["n"] == 3
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and good["sealed"] is True
        and good["poll_set"] is False
        and empty.get("ok") is False
        and empty.get("why") == "empty"
    )
    return {
        "ok": ok,
        "us": us,
        "n": good.get("n"),
        "tip": good.get("tip"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "mint": False,
        "socket": False,
    }


if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
