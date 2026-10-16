"""Link a trit onto the trit-pack chain close tip. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from trit_pack_chain_close import close_chain
from path import path
from validate import make, valid
from base_rate import BASE


def trit_pack_chain_close_link(sessions=None, actor="dev", hits=None):
    if sessions is None:
        sessions = [("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])]
    if hits is None:
        hits = ["field", "node"]
    t0 = time.perf_counter()
    closed = close_chain(sessions)
    if not closed.get("ok"):
        return {"ok": False, "why": closed.get("why", "close"), "mint": False, "socket": False}
    tip = closed["tip"]
    row = path(actor, hits, True)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why", "path"), "mint": False, "socket": False}
    text = "|".join(hits) + "|close_link|0.05"
    block = make(text, tip)
    check = valid(block)
    held = (
        row.get("released", 1) == 0
        and row.get("rate") == BASE
        and closed.get("held") is True
        and closed.get("rate") == BASE
        and closed.get("released") == 0
    )
    good = (
        check["ok"]
        and held
        and closed.get("mint") is False
        and closed.get("socket") is False
        and closed.get("poll_set") is False
        and closed.get("closed") is True
        and closed.get("packed_once") is True
        and block["prev"] == tip
    )
    return {
        "ok": good,
        "n": closed.get("n", 0) + 1,
        "tip": block["hash"],
        "prev": tip,
        "juniors": round(closed.get("juniors", 0) + row.get("juniors", 0), 4),
        "rate": BASE,
        "held": held,
        "released": 0,
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
        "closed": True,
        "linked": True,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
    }


def guard():
    t0 = time.perf_counter()
    good = trit_pack_chain_close_link()
    bad = trit_pack_chain_close_link(sessions=[("dev", ["gaia"]), ("", ["prompt"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        good["ok"]
        and good["n"] >= 4
        and good["rate"] == 0.05
        and good["held"] is True
        and good["released"] == 0
        and good["mint"] is False
        and good["socket"] is False
        and good["poll_set"] is False
        and good["closed"] is True
        and good["linked"] is True
        and bad["ok"] is False
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
        "poll_set": False,
        "closed": good.get("closed"),
        "linked": good.get("linked"),
    }


if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
