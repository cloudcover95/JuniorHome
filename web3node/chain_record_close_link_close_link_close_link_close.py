"""Close the chain record close link close link close link tip. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from chain_record_close_link_close_link_close_link import chain_record_close_link_close_link_close_link
from validate import make, valid
from base_rate import BASE


def chain_record_close_link_close_link_close_link_close(sessions=None, actor="dev", hits=None):
    t0 = time.perf_counter()
    linked = chain_record_close_link_close_link_close_link(sessions, actor, hits)
    if not linked.get("ok"):
        return {"ok": False, "why": linked.get("why", "close_link_close_link_close_link"), "mint": False, "socket": False}
    tip = linked["tip"]
    text = "close|close_link_close_link_close_link|" + str(linked.get("n", 0)) + "|0.05|held"
    block = make(text, tip)
    check = valid(block)
    good = (
        check["ok"]
        and linked.get("held") is True
        and linked.get("rate") == BASE
        and linked.get("released") == 0
        and linked.get("mint") is False
        and linked.get("socket") is False
        and linked.get("poll_set") is False
        and linked.get("closed") is True
        and linked.get("linked") is True
        and linked.get("in_pack") is True
        and linked.get("close_link_closed") is True
        and linked.get("close_link_close_linked") is True
        and linked.get("close_link_close_link_closed") is True
        and linked.get("close_link_close_link_close_linked") is True
        and block["prev"] == tip
    )
    return {
        "ok": good,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
        "n": linked.get("n", 0),
        "tip": block["hash"],
        "prev": tip,
        "rate": BASE,
        "juniors": linked.get("juniors"),
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
    }


if __name__ == "__main__":
    import json
    row = chain_record_close_link_close_link_close_link_close()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
