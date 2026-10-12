"""Link closes on the trit pack. Pool held at base rate. One pack per close."""
import sys
import time
from base_rate import BASE
from path import path
from validate import make, valid


def link(sessions):
    prev = "0" * 16
    blocks = []
    juniors = 0.0
    for actor, hits in sessions:
        row = path(actor, hits, True)
        if not row["ok"]:
            return {"ok": False, "why": row.get("why"), "n": len(blocks)}
        block = make(row["hash"], prev)
        check = valid(block)
        if not check["ok"]:
            return {"ok": False, "why": "trit", "n": len(blocks)}
        if row.get("rate") != BASE or row.get("released", -1) != 0:
            return {"ok": False, "why": "held", "n": len(blocks)}
        if row.get("mint") is not False or row.get("socket") is not False:
            return {"ok": False, "why": "flags", "n": len(blocks)}
        blocks.append(block["hash"])
        prev = block["hash"]
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
    }


def guard():
    t0 = time.perf_counter()
    good = link([("dev", ["gaia", "home"]), ("forge", ["field"]), ("quill", ["deck"])])
    bad = link([("dev", ["gaia"]), ("", ["prompt"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good["n"] == 3
        and good["held"] is True
        and good["rate"] == 0.05
        and good["released"] == 0
        and good["mint"] is False
        and good["socket"] is False
        and bad["ok"] is False
        and bad["why"] == "bot"
    )
    return {
        "ok": ok,
        "us": us,
        "n": good.get("n"),
        "tip": good.get("tip"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "juniors": good.get("juniors"),
        "mint": False,
        "socket": False,
    }


if __name__ == "__main__":
    import json

    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
