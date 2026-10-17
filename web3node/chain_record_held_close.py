"""Close the held chain record guard tip. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from chain_record_held_guard import guard
from validate import make, valid
from base_rate import BASE


def close():
    t0 = time.perf_counter()
    guarded = guard()
    if not guarded.get("ok"):
        return {"ok": False, "why": "guard", "mint": False, "socket": False}
    tip = guarded["tip"]
    text = "held_close|" + str(guarded.get("n", 0)) + "|0.05|held"
    block = make(text, tip)
    check = valid(block)
    good = (
        check["ok"]
        and guarded.get("held") is True
        and guarded.get("rate") == BASE
        and guarded.get("released") == 0
        and guarded.get("mint") is False
        and guarded.get("socket") is False
        and guarded.get("in_pack") is True
        and guarded.get("guarded") is True
        and block["prev"] == tip
    )
    return {
        "ok": good,
        "us": round((time.perf_counter() - t0) * 1e6, 1),
        "n": guarded.get("n", 0),
        "tip": block["hash"],
        "prev": tip,
        "rate": BASE,
        "juniors": guarded.get("juniors"),
        "released": 0,
        "mint": False,
        "socket": False,
        "held": True,
        "in_pack": True,
        "base": BASE,
        "closed": True,
        "guarded": True,
    }


if __name__ == "__main__":
    import json
    out = close()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] else 1)
