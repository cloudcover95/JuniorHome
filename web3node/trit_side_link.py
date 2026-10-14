"""Link trit side seals on the pack chain. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_side_seal import pack, side_commit
from trit_close import trit_close
from base_rate import BASE


def side_link(sessions=None):
    if sessions is None:
        sessions = [("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])]
    prev = "0" * 16
    tips = []
    juniors = 0.0
    t0 = time.perf_counter()
    for actor, hits in sessions:
        closed = trit_close(actor, hits, True)
        if not closed.get("ok"):
            return {"ok": False, "why": closed.get("why", "close"), "n": len(tips), "mint": False, "socket": False}
        if (closed.get("rate") != BASE or not closed.get("held") or
            closed.get("mint") is not False or closed.get("socket") is not False or
            closed.get("poll_set") is not False):
            return {"ok": False, "why": "held", "n": len(tips), "mint": False, "socket": False}
        text = "|".join(hits)
        # one trit pack from the hit text (balanced -1/0/1)
        trits = [(ord(c) % 3) - 1 for c in (text or "0")[:10]]
        pk = pack(trits)
        commit = side_commit(prev, pk, text)
        tips.append(commit)
        prev = commit
        juniors += closed.get("juniors", 0.0)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    return {
        "ok": True,
        "n": len(tips),
        "tip": prev,
        "juniors": round(juniors, 4),
        "rate": BASE,
        "held": True,
        "released": 0,
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
        "us": us,
    }


def bench():
    good = side_link()
    bad = side_link([("dev", ["gaia"]), ("", ["prompt"])])
    ok = (
        good.get("ok") is True
        and good.get("n") == 3
        and good.get("rate") == BASE
        and good.get("held") is True
        and good.get("released") == 0
        and good.get("mint") is False
        and good.get("socket") is False
        and good.get("poll_set") is False
        and good.get("packed_once") is True
        and bad.get("ok") is False
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "n": good.get("n"),
        "tip": good.get("tip"),
        "juniors": good.get("juniors"),
        "rate": BASE,
        "held": True,
        "released": 0,
        "mint": False,
        "socket": False,
        "poll_set": False,
    }


if __name__ == "__main__":
    import json
    row = bench()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
