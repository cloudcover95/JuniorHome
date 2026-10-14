"""Bench the trit pack chain. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_pack_chain import trit_pack_chain
from base_rate import BASE

def bench(rounds=3):
    t0 = time.perf_counter()
    rows = []
    for _ in range(rounds):
        row = trit_pack_chain()
        rows.append(row)
        if not row.get("ok"):
            return {"ok": False, "why": row.get("why", "chain"), "mint": False, "socket": False}
    us = round((time.perf_counter() - t0) * 1e6, 1)
    mean_us = round(us / max(rounds, 1), 1)
    held_ok = all(
        r.get("ok") is True
        and r.get("rate") == BASE
        and r.get("held") is True
        and r.get("released") == 0
        and r.get("mint") is False
        and r.get("socket") is False
        and r.get("poll_set") is False
        and r.get("n") == 3
        for r in rows
    )
    good = len(rows) == rounds and held_ok
    return {
        "ok": good,
        "us": us,
        "mean_us": mean_us,
        "rounds": rounds,
        "tip": rows[-1].get("tip") if rows else None,
        "juniors": rows[-1].get("juniors") if rows else 0,
        "rate": BASE,
        "held": held_ok,
        "released": 0,
        "mint": False,
        "socket": False,
        "poll_set": False,
    }

def guard():
    good = bench(3)
    bad = trit_pack_chain([("dev", ["gaia"]), ("", ["prompt"])])
    ok = (
        good["ok"]
        and good["rounds"] == 3
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and good["poll_set"] is False
        and bad.get("ok") is False
        and bad.get("why") == "bot"
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "mean_us": good.get("mean_us"),
        "rounds": good.get("rounds"),
        "tip": good.get("tip"),
        "juniors": good.get("juniors"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "mint": False,
        "socket": False,
    }

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
