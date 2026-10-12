"""Guard the trit-pack chain. Opt-in AntiCon one pack, base 0.05 held, mint false, socket false. No poll rate set."""
import sys
import time
from trit_multi import trit_multi
from trit_link import trit_link
from base_rate import BASE

def trit_chain_guard(sessions=None):
    if sessions is None:
        sessions = [("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])]
    t0 = time.perf_counter()
    multi = trit_multi(sessions)
    if not multi["ok"]:
        return {"ok": False, "why": multi.get("why", "multi"), "mint": False, "socket": False}
    link = trit_link("dev", ["gaia", "home"], True)
    if not link["ok"]:
        return {"ok": False, "why": link.get("why", "link"), "mint": False, "socket": False}
    held = (
        multi.get("held") is True
        and multi.get("rate") == BASE
        and multi.get("released") == 0
        and link.get("held") is True
        and link.get("rate") == BASE
    )
    good = (
        multi["ok"]
        and link["ok"]
        and multi["n"] == len(sessions)
        and held
        and multi["mint"] is False
        and multi["socket"] is False
        and multi.get("poll_set") is False
        and multi.get("packed_once") is True
        and link["mint"] is False
        and link["socket"] is False
        and link.get("poll_set") is False
        and link.get("packed_once") is True
        and link.get("prev") == link.get("close_hash")
    )
    return {
        "ok": good,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
        "n": multi["n"],
        "tip": multi.get("tip"),
        "link": link.get("link"),
        "rate": BASE,
        "held": held,
        "released": 0,
        "juniors": multi.get("juniors"),
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
    }

def guard():
    good = trit_chain_guard()
    bad = trit_chain_guard([("dev", ["gaia"]), ("", ["prompt"])])
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
    return {
        "ok": ok,
        "us": good.get("us"),
        "n": good.get("n"),
        "tip": good.get("tip"),
        "link": good.get("link"),
        "juniors": good.get("juniors"),
        "rate": good.get("rate"),
    }

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
