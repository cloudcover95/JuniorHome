"""Bench the sealed trit-pack chain. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll rate set."""
import sys
import time
from trit_chain_seal import trit_chain_seal
from base_rate import BASE

def trit_seal_bench(rounds=5, sessions=None):
    if sessions is None:
        sessions = [("dev", ["gaia", "home"]), ("home", ["prompt"]), ("pi", ["field"])]
    t0 = time.perf_counter()
    rows = []
    for _ in range(rounds):
        row = trit_chain_seal(sessions)
        rows.append(row)
        if not row["ok"]:
            return {"ok": False, "why": row.get("why", "seal"), "mint": False, "socket": False}
    us = round((time.perf_counter() - t0) * 1e6, 1)
    mean_us = round(sum(r.get("us", 0) for r in rows) / rounds, 1)
    seals = {r.get("seal") for r in rows}
    tips = {r.get("tip") for r in rows}
    held_ok = all(r.get("held") is True and r.get("rate") == BASE and r.get("released") == 0 for r in rows)
    good = (
        len(rows) == rounds
        and held_ok
        and all(r["mint"] is False and r["socket"] is False and r.get("poll_set") is False for r in rows)
        and all(r.get("packed_once") is True for r in rows)
        and len(seals) == 1
        and len(tips) == 1
    )
    return {
        "ok": good,
        "us": us,
        "mean_us": mean_us,
        "rounds": rounds,
        "n": rows[0].get("n") if rows else 0,
        "tip": rows[0].get("tip") if rows else None,
        "seal": rows[0].get("seal") if rows else None,
        "rate": BASE,
        "held": held_ok,
        "released": 0,
        "juniors": rows[0].get("juniors") if rows else None,
        "mint": False,
        "socket": False,
        "poll_set": False,
        "packed_once": True,
    }

def guard():
    good = trit_seal_bench(5)
    bad = trit_seal_bench(1, [("dev", ["gaia"]), ("", ["prompt"])])
    ok = (
        good["ok"]
        and good["rounds"] == 5
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
        "mean_us": good.get("mean_us"),
        "rounds": good.get("rounds"),
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
