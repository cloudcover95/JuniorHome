"""Chain successive close-links. Tip advances. Pool stays held. Opt-in, AntiCon, one trit pack, base rate 0.05, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_close_link import close_link


def close_link_chain(sessions):
    prev = "0" * 16
    rows = []
    for actor, hits in sessions:
        row = close_link(actor, hits, prev)
        if not row.get("ok"):
            return {"ok": False, "why": row.get("why", "close"), "n": len(rows), "mint": False, "socket": False}
        if row.get("rate") != 0.05 or row.get("released") != 0:
            return {"ok": False, "why": "held", "n": len(rows), "mint": False, "socket": False}
        if row.get("mint") is not False or row.get("socket") is not False:
            return {"ok": False, "why": "flags", "n": len(rows), "mint": False, "socket": False}
        if not row.get("packed_once") or not row.get("in_pack"):
            return {"ok": False, "why": "pack", "n": len(rows), "mint": False, "socket": False}
        rows.append(row)
        prev = row["hash"]
    return {
        "ok": True,
        "n": len(rows),
        "tip": prev,
        "rate": 0.05,
        "released": 0,
        "held": True,
        "mint": False,
        "socket": False,
        "packed_once": True,
        "in_pack": True,
        "hashes": [r["hash"] for r in rows],
    }


def bench():
    t0 = time.perf_counter()
    good = close_link_chain([("dev", ["gaia"]), ("home", ["prompt"]), ("pi", ["field"])])
    bad = close_link_chain([("dev", ["gaia"]), ("", ["prompt"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good["n"] == 3
        and good["tip"] == good["hashes"][-1]
        and good["rate"] == 0.05
        and good["released"] == 0
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and good["packed_once"] is True
        and bad["ok"] is False
        and bad.get("why") == "bot"
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
