"""Bench and guard the trit-close link. Opt-in, AntiCon, one trit pack per close.
Base rate 0.05 held, pool held, mint false, socket false.
No poll sets the rate.
"""
import sys
import time
from base_rate import BASE
from trit_close_link import link_closes


def bench(rounds=4):
    t0 = time.perf_counter()
    sessions = [
        ("dev", ["gaia", "home"]),
        ("home", ["field", "deck"]),
        ("forge", ["gaia"]),
        ("quill", ["deck", "home"]),
    ]
    rows = []
    for i in range(rounds):
        row = link_closes(sessions[: 2 + (i % 2)])
        if not row.get("ok"):
            return {"ok": False, "why": row.get("why"), "i": i, "mint": False, "socket": False}
        rows.append(row)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    mean_us = round(us / max(rounds, 1), 1)
    tips = {r.get("tip") for r in rows}
    held_ok = all(
        r.get("held") is True
        and r.get("rate") == BASE
        and r.get("released") == 0
        and r.get("mint") is False
        and r.get("socket") is False
        and r.get("pool_held") is True
        for r in rows
    )
    return {
        "ok": held_ok and len(tips) >= 1,
        "rounds": rounds,
        "us": us,
        "mean_us": mean_us,
        "tips": len(tips),
        "rate": BASE,
        "released": 0,
        "held": True,
        "mint": False,
        "socket": False,
        "pool_held": True,
        "n": rows[-1].get("n") if rows else 0,
    }


def guard():
    good = bench(3)
    bad_bot = link_closes([("", ["gaia"])])
    bad_empty = link_closes([])
    ok = (
        good["ok"]
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and good["pool_held"] is True
        and bad_bot.get("ok") is False
        and bad_bot.get("why") == "bot"
        and bad_empty.get("ok") is True  # empty chain is vacuously held
        and bad_empty.get("n") == 0
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "rounds": good.get("rounds"),
        "mean_us": good.get("mean_us"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "n": good.get("n"),
        "mint": False,
        "socket": False,
        "pool_held": True,
    }


if __name__ == "__main__":
    import json

    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
