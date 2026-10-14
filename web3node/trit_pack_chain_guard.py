"""Guard the trit pack chain. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_pack_chain import trit_pack_chain
from validate import make, valid
from base_rate import BASE


def guard():
    t0 = time.perf_counter()
    good = trit_pack_chain()
    bad_actor = trit_pack_chain([("dev", ["gaia"]), ("", ["prompt"])])
    # tamper tip and confirm valid denies
    if good.get("ok"):
        tip = good["tip"]
        tampered = make("tamper", tip)
        tampered["hash"] = "0" * 16
        deny = valid(tampered)
    else:
        deny = {"ok": True}
    us = round((time.perf_counter() - t0) * 1e6, 1)
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
        and bad_actor.get("ok") is False
        and bad_actor.get("why") == "bot"
        and deny.get("ok") is False
    )
    return {
        "ok": ok,
        "us": us,
        "n": good.get("n"),
        "tip": good.get("tip"),
        "juniors": good.get("juniors"),
        "rate": BASE,
        "held": True,
        "released": 0,
        "mint": False,
        "socket": False,
        "poll_set": False,
        "why_bad": bad_actor.get("why"),
    }


if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
