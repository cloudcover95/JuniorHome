"""Guard a linked trit-pack close. Opt-in, AntiCon, one trit pack, base 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from path import path
from validate import make, valid
from base_rate import BASE


def link_close(sessions):
    if not sessions:
        return {"ok": False, "why": "empty"}
    prev = "0" * 16
    blocks = []
    juniors = 0.0
    for actor, hits in sessions:
        row = path(actor, hits, True)
        if not row.get("ok"):
            return {"ok": False, "why": row.get("why"), "n": len(blocks)}
        if row.get("rate") != BASE or row.get("released") != 0:
            return {"ok": False, "why": "rate", "n": len(blocks)}
        if row.get("mint") is not False or row.get("socket") is not False:
            return {"ok": False, "why": "flags", "n": len(blocks)}
        block = make("|" .join(hits), prev)
        check = valid(block)
        if not check["ok"]:
            return {"ok": False, "why": "trit", "n": len(blocks)}
        blocks.append(check["hash"])
        prev = check["hash"]
        juniors += row.get("juniors", 0.0)
    return {
        "ok": True,
        "n": len(blocks),
        "tip": prev,
        "juniors": round(juniors, 4),
        "rate": BASE,
        "released": 0,
        "held": True,
        "mint": False,
        "socket": False,
        "poll_set": False,
    }


def guard():
    t0 = time.perf_counter()
    good = link_close([("dev", ["gaia", "home"]), ("home", ["prompt"])])
    bad_iso = path("dev", ["gaia"], False)
    bad_bot = path("", ["gaia"], True)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good["n"] == 2
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and bad_iso.get("ok") is False
        and bad_iso.get("why") == "isolated"
        and bad_bot.get("ok") is False
        and bad_bot.get("why") == "bot"
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
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
