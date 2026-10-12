"""Bench the path/record trit-pack guard. Base rate 0.05 held."""
import sys
import time
from path_record_guard import path_record_guard
from base_rate import BASE

def path_record_bench(rounds=3):
    t0 = time.perf_counter()
    rows = []
    prev = "0" * 16
    for _ in range(rounds):
        row = path_record_guard(hits=["field", "koirig"], prev=prev)
        rows.append(row)
        if not row["ok"]:
            return {"ok": False, "why": row.get("why", "guard"), "mint": False, "socket": False}
        prev = row["hash"]
    us = round((time.perf_counter() - t0) * 1e6, 1)
    mean_us = round(sum(r.get("us", 0) for r in rows) / rounds, 1) if rows else 0
    held_ok = all(r.get("held") is True and r.get("rate") == BASE and r.get("released") == 0 for r in rows)
    good = (
        len(rows) == rounds
        and held_ok
        and all(r["mint"] is False and r["socket"] is False for r in rows)
        and all(r.get("in_pack") is True for r in rows)
    )
    return {
        "ok": good,
        "us": us,
        "mean_us": mean_us,
        "rounds": rounds,
        "tip": rows[-1].get("hash") if rows else None,
        "rate": BASE,
        "held": held_ok,
        "released": 0,
        "juniors": rows[0].get("juniors") if rows else None,
        "mint": False,
        "socket": False,
        "in_pack": True,
    }

def guard():
    good = path_record_bench(3)
    bad_row = path_record_guard("", ["deck"])
    ok = (
        good["ok"]
        and good["rounds"] == 3
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and bad_row["ok"] is False
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "rounds": good.get("rounds"),
        "tip": good.get("tip"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "juniors": good.get("juniors"),
        "mint": False,
        "socket": False,
    }

if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
