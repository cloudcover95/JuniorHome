"""Link a trit close onto the chain record tip. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from chain_record import run
from path import path
from validate import make, valid
from base_rate import BASE


def chain_record_link(sessions=None, actor="dev", hits=None):
    if sessions is None:
        sessions = [("dev", ["gaia"]), ("home", ["prompt"])]
    if hits is None:
        hits = ["field", "node"]
    t0 = time.perf_counter()
    recorded = run(sessions)
    if not recorded["ok"]:
        return {"ok": False, "why": recorded.get("why", "record"), "mint": False, "socket": False}
    tip = recorded["tip"]
    row = path(actor, hits, True)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why", "path"), "mint": False, "socket": False}
    text = "|".join(hits) + "|link|0.05"
    block = make(text, tip)
    check = valid(block)
    held = row.get("released", 1) == 0 and row.get("rate") == BASE
    good = (
        check["ok"]
        and recorded["n"] == len(sessions)
        and recorded.get("in_pack") is True
        and recorded.get("released") == 0
        and recorded.get("mint") is False
        and recorded.get("socket") is False
        and held
        and row["mint"] is False
        and row["socket"] is False
        and block["prev"] == tip
    )
    return {
        "ok": good,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
        "n": recorded["n"] + 1,
        "tip": block["hash"],
        "prev": tip,
        "rate": BASE,
        "juniors": round(recorded.get("juniors", 0) + row.get("juniors", 0), 4),
        "released": 0,
        "mint": False,
        "socket": False,
        "held": True,
        "in_pack": True,
        "poll_set": False,
    }


if __name__ == "__main__":
    import json
    row = chain_record_link()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
