"""Seal the guarded trit-pack chain. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll rate set."""
import sys
import time
from trit_chain_guard import trit_chain_guard
from validate import make, valid
from base_rate import BASE

def trit_chain_seal(sessions=None):
    if sessions is None:
        sessions = [("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])]
    t0 = time.perf_counter()
    guarded = trit_chain_guard(sessions)
    if not guarded["ok"]:
        return {"ok": False, "why": guarded.get("why", "guard"), "mint": False, "socket": False}
    tip = guarded.get("tip") or ("0" * 16)
    text = "seal|0.05|held"
    block = make(text, tip)
    check = valid(block)
    held = (
        guarded.get("held") is True
        and guarded.get("rate") == BASE
        and guarded.get("released") == 0
    )
    good = (
        check["ok"]
        and held
        and guarded["mint"] is False
        and guarded["socket"] is False
        and guarded.get("poll_set") is False
        and guarded.get("packed_once") is True
        and block["prev"] == tip
    )
    return {
        "ok": good,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
        "n": guarded.get("n"),
        "tip": tip,
        "seal": block["hash"],
        "prev": tip,
        "rate": BASE,
        "held": held,
        "released": 0,
        "juniors": guarded.get("juniors"),
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
    }

def guard():
    good = trit_chain_seal()
    bad = trit_chain_seal([("dev", ["gaia"]), ("", ["prompt"])])
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
        "seal": good.get("seal"),
        "juniors": good.get("juniors"),
        "rate": good.get("rate"),
    }

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
