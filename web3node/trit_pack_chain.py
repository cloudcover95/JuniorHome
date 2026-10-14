"""Trit pack chain. Multiple closes linked by tip. Opt-in, AntiCon, one trit pack each, base rate 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_close import trit_close
from validate import make, valid
from base_rate import BASE

def trit_pack_chain(sessions=None):
    if sessions is None:
        sessions = [("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])]
    prev = "0" * 16
    blocks = []
    juniors = 0.0
    t0 = time.perf_counter()
    for actor, hits in sessions:
        closed = trit_close(actor, hits, True)
        if not closed.get("ok"):
            return {"ok": False, "why": closed.get("why", "close"), "n": len(blocks), "mint": False, "socket": False}
        if closed.get("rate") != BASE or not closed.get("held"):
            return {"ok": False, "why": "held", "n": len(blocks), "mint": False, "socket": False}
        if closed.get("mint") is not False or closed.get("socket") is not False or closed.get("poll_set") is not False:
            return {"ok": False, "why": "flags", "n": len(blocks), "mint": False, "socket": False}
        text = "|".join(hits) + "|pack|0.05"
        block = make(text, prev)
        check = valid(block)
        if not check["ok"]:
            return {"ok": False, "why": "trit", "n": len(blocks), "mint": False, "socket": False}
        blocks.append(block["hash"])
        prev = block["hash"]
        juniors += closed.get("juniors", 0.0)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    return {
        "ok": True,
        "n": len(blocks),
        "tip": prev,
        "juniors": round(juniors, 4),
        "rate": BASE,
        "held": True,
        "released": 0,
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
        "us": us,
    }

def guard():
    good = trit_pack_chain()
    bad = trit_pack_chain([("dev", ["gaia"]), ("", ["prompt"])])
    ok = (
        good["ok"]
        and good["n"] == 3
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
        "us": good.get("us"),
        "n": good.get("n"),
        "tip": good.get("tip"),
        "juniors": good.get("juniors"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "mint": False,
        "socket": False,
    }

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
