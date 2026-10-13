"""Close the guarded trit-pack link into one final pack. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_guard_link import link
from base_rate import BASE
from validate import make, valid


def close(sessions):
    row = link(sessions)
    if not row.get("ok"):
        return {"ok": False, "why": row.get("why", "link"), "n": row.get("n", 0)}
    tip = row.get("tip") or ""
    block = make(tip, tip)
    check = valid(block)
    if not check["ok"]:
        return {"ok": False, "why": "close", "n": row.get("n")}
    if row.get("rate") != BASE or row.get("released") != 0 or not row.get("held"):
        return {"ok": False, "why": "held", "n": row.get("n")}
    if row.get("mint") is not False or row.get("socket") is not False:
        return {"ok": False, "why": "flags", "n": row.get("n")}
    return {
        "ok": True,
        "n": row.get("n"),
        "tip": block["hash"],
        "prev": tip,
        "juniors": row.get("juniors"),
        "rate": BASE,
        "released": 0,
        "held": True,
        "mint": False,
        "socket": False,
        "avg_trit": row.get("avg_trit"),
        "closed": True,
    }


def guard():
    t0 = time.perf_counter()
    sessions = [("dev", ["gaia", "home"]), ("forge", ["field"]), ("quill", ["deck"])]
    good = close(sessions)
    bad = close([("dev", ["gaia"]), ("", ["prompt"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good.get("n") == 3
        and good.get("closed") is True
        and good.get("held") is True
        and good.get("rate") == 0.05
        and good.get("released") == 0
        and good.get("mint") is False
        and good.get("socket") is False
        and bad.get("ok") is False
        and bad.get("why") == "bot"
    )
    return {
        "ok": ok,
        "us": us,
        "n": good.get("n"),
        "tip": good.get("tip"),
        "prev": good.get("prev"),
        "juniors": good.get("juniors"),
        "avg_trit": good.get("avg_trit"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "mint": False,
        "socket": False,
        "closed": good.get("closed"),
    }


if __name__ == "__main__":
    import json

    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
