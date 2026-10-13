"""Bench the guarded trit-pack link. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_guard_link import link
from base_rate import BASE


def trit_guard_link_bench(rounds=3):
    t0 = time.perf_counter()
    rows = []
    sessions = [("dev", ["gaia", "home"]), ("forge", ["field"]), ("quill", ["deck"])]
    for _ in range(rounds):
        row = link(sessions)
        rows.append(row)
        if not row.get("ok"):
            return {"ok": False, "why": row.get("why", "link"), "mint": False, "socket": False}
    us = round((time.perf_counter() - t0) * 1e6, 1)
    mean_us = round(sum(r.get("us", 0) for r in rows) / max(rounds, 1), 1) if rows else 0
    tips = {r.get("tip") for r in rows}
    held_ok = all(
        r.get("held") is True
        and r.get("rate") == BASE
        and r.get("released") == 0
        and r.get("mint") is False
        and r.get("socket") is False
        for r in rows
    )
    good = (
        len(rows) == rounds
        and held_ok
        and all(r.get("n") == 3 for r in rows)
        and len(tips) == 1
    )
    return {
        "ok": good,
        "us": us,
        "mean_us": mean_us,
        "rounds": rounds,
        "n": rows[0].get("n") if rows else 0,
        "tip": rows[0].get("tip") if rows else None,
        "juniors": rows[0].get("juniors") if rows else None,
        "avg_trit": rows[0].get("avg_trit") if rows else None,
        "rate": BASE,
        "held": held_ok,
        "released": 0,
        "mint": False,
        "socket": False,
    }


def guard():
    good = trit_guard_link_bench(3)
    bad = link([("dev", ["gaia"]), ("", ["prompt"])])
    ok = (
        good["ok"]
        and good["rounds"] == 3
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and bad.get("ok") is False
        and bad.get("why") == "bot"
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "mean_us": good.get("mean_us"),
        "rounds": good.get("rounds"),
        "n": good.get("n"),
        "tip": good.get("tip"),
        "juniors": good.get("juniors"),
        "avg_trit": good.get("avg_trit"),
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
