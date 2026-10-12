"""Multi trit-pack chain. Each close links the prior tip. One pack, base 0.05 held, no mint, no socket."""
import sys
import time
from trit_close import trit_close
from validate import make, valid
from base_rate import BASE

def trit_multi(sessions):
    prev = "0" * 16
    blocks = []
    juniors = 0.0
    for actor, hits in sessions:
        closed = trit_close(actor, hits, True)
        if not closed["ok"] or closed.get("rate") != BASE or not closed.get("held"):
            return {"ok": False, "why": closed.get("why", "close"), "n": len(blocks),
                    "mint": False, "socket": False}
        text = "|".join(hits) + "|link|0.05"
        block = make(text, prev)
        check = valid(block)
        if not check["ok"]:
            return {"ok": False, "why": "trit", "n": len(blocks), "mint": False, "socket": False}
        blocks.append(block["hash"])
        prev = block["hash"]
        juniors += closed.get("juniors", 0.0)
    return {"ok": True, "n": len(blocks), "tip": prev, "juniors": round(juniors, 4),
            "rate": BASE, "held": True, "released": 0, "mint": False, "socket": False,
            "poll_set": False, "packed_once": True}

def guard():
    t0 = time.perf_counter()
    good = trit_multi([("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])])
    bad = trit_multi([("dev", ["gaia"]), ("", ["prompt"])])
    us = round((time.perf_counter() - t0) * 1e6, 1)
    ok = (good["ok"] and good["n"] == 3 and good["rate"] == 0.05
          and good["held"] is True and good["released"] == 0
          and good["mint"] is False and good["socket"] is False
          and good["poll_set"] is False
          and bad["ok"] is False and bad.get("why") == "bot")
    return {"ok": ok, "us": us, "n": good.get("n"), "tip": good.get("tip"),
            "juniors": good.get("juniors")}

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
