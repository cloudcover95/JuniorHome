"""Link a trit onto the chain record close link close link close link close link close tip. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from chain_record_close_link_close_link_close_link_close_link_close import chain_record_close_link_close_link_close_link_close_link_close
from path import path
from validate import make, valid
from base_rate import BASE


def chain_record_close_link_close_link_close_link_close_link_close_link(sessions=None, actor="dev", hits=None):
    if sessions is None:
        sessions = [("dev", ["gaia"]), ("home", ["prompt"])]
    if hits is None:
        hits = ["field", "node"]
    t0 = time.perf_counter()
    closed = chain_record_close_link_close_link_close_link_close_link_close(sessions, actor, hits)
    if not closed.get("ok"):
        return {"ok": False, "why": closed.get("why", "close_link_close_link_close_link_close_link_close"), "mint": False, "socket": False}
    tip = closed["tip"]
    row = path(actor, hits, True)
    if not row["ok"]:
        return {"ok": False, "why": row.get("why", "path"), "mint": False, "socket": False}
    text = "|".join(hits) + "|close_link_close_link_close_link_close_link_close_link|0.05"
    block = make(text, tip)
    check = valid(block)
    held = row.get("released", 1) == 0 and row.get("rate") == BASE
    good = (
        check["ok"]
        and closed.get("held") is True
        and closed.get("rate") == BASE
        and closed.get("released") == 0
        and closed.get("mint") is False
        and closed.get("socket") is False
        and closed.get("poll_set") is False
        and closed.get("closed") is True
        and closed.get("linked") is True
        and closed.get("in_pack") is True
        and closed.get("close_link_closed") is True
        and closed.get("close_link_close_linked") is True
        and closed.get("close_link_close_link_closed") is True
        and closed.get("close_link_close_link_close_linked") is True
        and closed.get("close_link_close_link_close_link_closed") is True
        and closed.get("close_link_close_link_close_link_close_linked") is True
        and closed.get("close_link_close_link_close_link_close_link_closed") is True
        and held
        and row["mint"] is False
        and row["socket"] is False
        and block["prev"] == tip
    )
    return {
        "ok": good,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
        "n": closed.get("n", 0) + 1,
        "tip": block["hash"],
        "prev": tip,
        "rate": BASE,
        "juniors": round(closed.get("juniors", 0) + row.get("juniors", 0), 4),
        "released": 0,
        "mint": False,
        "socket": False,
        "held": True,
        "in_pack": True,
        "poll_set": False,
        "closed": True,
        "linked": True,
        "close_link_closed": True,
        "close_link_close_linked": True,
        "close_link_close_link_closed": True,
        "close_link_close_link_close_linked": True,
        "close_link_close_link_close_link_closed": True,
        "close_link_close_link_close_link_close_linked": True,
        "close_link_close_link_close_link_close_link_closed": True,
        "close_link_close_link_close_link_close_link_close_linked": True,
    }


if __name__ == "__main__":
    import json
    row = chain_record_close_link_close_link_close_link_close_link_close_link()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
