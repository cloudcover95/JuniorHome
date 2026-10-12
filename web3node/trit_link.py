"""Link the trit close on the path tip. One trit pack, base rate 0.05 held, mint false, socket false."""
import sys
import time
from trit_close import trit_close
from validate import make, valid
from base_rate import BASE

def trit_link(actor="dev", hits=None, opted_in=True):
    if hits is None:
        hits = ["gaia", "home"]
    t0 = time.perf_counter()
    closed = trit_close(actor, hits, opted_in)
    if not closed["ok"]:
        return {"ok": False, "why": closed.get("why", "close"), "mint": False, "socket": False}
    text = "|".join(hits) + "|link|0.05"
    block = make(text, closed["hash"])
    check = valid(block)
    held = closed.get("held", False) and closed.get("rate") == BASE
    good = (
        check["ok"]
        and closed["packed_once"]
        and held
        and closed["mint"] is False
        and closed["socket"] is False
        and closed.get("poll_set") is False
        and block["prev"] == closed["hash"]
    )
    return {
        "ok": good,
        "rate": BASE,
        "held": held,
        "link": block["hash"],
        "prev": closed["hash"],
        "close_hash": closed["hash"],
        "packed_once": True,
        "juniors": closed.get("juniors"),
        "mint": False,
        "socket": False,
        "poll_set": False,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
    }

if __name__ == "__main__":
    import json
    row = trit_link()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
