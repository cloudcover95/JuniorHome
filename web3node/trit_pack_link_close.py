"""Close the trit-pack link. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_pack_link import link
from base_rate import BASE
from validate import make, valid


def close_link(actor, hits, prev="0" * 16, opted_in=True):
    linked = link(actor, hits, prev, opted_in)
    if not linked.get("ok"):
        return {"ok": False, "why": linked.get("why", "link"), "linked": False, "closed": False}
    tip = linked.get("hash")
    block = make("|" .join(hits) + "|close", tip)
    check = valid(block)
    if not check["ok"]:
        return {"ok": False, "why": "close", "linked": True, "closed": False}
    if linked.get("rate") != BASE or linked.get("released") != 0:
        return {"ok": False, "why": "held", "linked": True, "closed": False}
    if linked.get("mint") is not False or linked.get("socket") is not False:
        return {"ok": False, "why": "flags", "linked": True, "closed": False}
    return {
        "ok": True,
        "hash": block["hash"],
        "prev": tip,
        "link_prev": prev,
        "link_hash": tip,
        "rate": BASE,
        "released": 0,
        "held": True,
        "mint": False,
        "socket": False,
        "linked": True,
        "closed": True,
    }


def guard():
    t0 = time.perf_counter()
    a = close_link("dev", ["gaia"], "0" * 16, True)
    b = close_link("home", ["deck"], a.get("hash"), True)
    bad = close_link("dev", ["gaia"], "0" * 16, False)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (
        a.get("ok")
        and b.get("ok")
        and b.get("link_prev") == a.get("hash")
        and a.get("closed") is True
        and b.get("closed") is True
        and a.get("rate") == 0.05
        and a.get("held") is True
        and a.get("mint") is False
        and a.get("socket") is False
        and bad.get("ok") is False
        and bad.get("why") == "isolated"
    )
    return {
        "ok": ok,
        "us": us,
        "tip": b.get("hash") if b.get("ok") else None,
        "rate": BASE,
        "held": True,
        "released": 0,
        "mint": False,
        "socket": False,
        "closed": True,
    }


if __name__ == "__main__":
    import json

    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
