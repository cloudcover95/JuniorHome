"""Link a sealed trit-pack tip. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll rate set."""
import sys
import time
from trit_chain_seal import trit_chain_seal
from trit_close import trit_close
from validate import make, valid
from base_rate import BASE

def trit_seal_link(actor="forge", hits=None, opted_in=True):
    if hits is None:
        hits = ["deck", "field"]
    t0 = time.perf_counter()
    sealed = trit_chain_seal()
    if not sealed["ok"]:
        return {"ok": False, "why": sealed.get("why", "seal"), "mint": False, "socket": False}
    closed = trit_close(actor, hits, opted_in)
    if not closed["ok"]:
        return {"ok": False, "why": closed.get("why", "close"), "mint": False, "socket": False}
    text = "|".join(hits) + "|seal_link|0.05"
    block = make(text, sealed["seal"])
    check = valid(block)
    held = (
        sealed.get("held") is True
        and sealed.get("rate") == BASE
        and sealed.get("released") == 0
        and closed.get("held") is True
        and closed.get("rate") == BASE
    )
    good = (
        check["ok"]
        and closed["packed_once"]
        and held
        and sealed["mint"] is False
        and sealed["socket"] is False
        and sealed.get("poll_set") is False
        and closed["mint"] is False
        and closed["socket"] is False
        and closed.get("poll_set") is False
        and block["prev"] == sealed["seal"]
    )
    return {
        "ok": good,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
        "n": sealed.get("n"),
        "tip": sealed.get("tip"),
        "seal": sealed.get("seal"),
        "link": block["hash"],
        "prev": sealed["seal"],
        "close_hash": closed.get("hash"),
        "rate": BASE,
        "held": held,
        "released": 0,
        "juniors": closed.get("juniors"),
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
    }

def guard():
    good = trit_seal_link()
    bad = trit_seal_link("", ["deck"])
    ok = (
        good["ok"]
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and good["poll_set"] is False
        and good.get("packed_once") is True
        and bad["ok"] is False
        and bad.get("why") in ("bot", "isolated", "path")
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "n": good.get("n"),
        "tip": good.get("tip"),
        "seal": good.get("seal"),
        "link": good.get("link"),
        "juniors": good.get("juniors"),
        "rate": good.get("rate"),
    }

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
