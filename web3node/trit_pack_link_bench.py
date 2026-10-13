"""Bench the trit-pack link. Opt-in, AntiCon, one trit pack, base rate 0.05 held, mint false, socket false. No poll sets rate."""
import sys
import time
from trit_pack_link import link
from base_rate import BASE


def trit_pack_link_bench(rounds=3):
    t0 = time.perf_counter()
    rows = []
    prev = "0" * 16
    for _ in range(rounds):
        row = link("dev", ["gaia", "home"], prev, True)
        rows.append(row)
        if not row.get("ok"):
            return {"ok": False, "why": row.get("why", "link"), "mint": False, "socket": False}
        prev = row.get("hash")
    us = round((time.perf_counter() - t0) * 1e6, 1)
    mean_us = round(sum(1 for _ in rows) and us / max(rounds, 1), 1)
    held_ok = all(
        r.get("ok") is True
        and r.get("rate") == BASE
        and r.get("released") == 0
        and r.get("mint") is False
        and r.get("socket") is False
        and r.get("linked") is True
        for r in rows
    )
    chain_ok = all(rows[i].get("prev") == (rows[i - 1].get("hash") if i else "0" * 16) for i in range(len(rows)))
    good = len(rows) == rounds and held_ok and chain_ok
    return {
        "ok": good,
        "us": us,
        "mean_us": mean_us,
        "rounds": rounds,
        "tip": rows[-1].get("hash") if rows else None,
        "rate": BASE,
        "held": held_ok,
        "released": 0,
        "mint": False,
        "socket": False,
    }


def guard():
    good = trit_pack_link_bench(3)
    bad = link("dev", ["gaia"], "0" * 16, False)
    ok = (
        good["ok"]
        and good["rounds"] == 3
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and bad.get("ok") is False
        and bad.get("why") == "isolated"
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "mean_us": good.get("mean_us"),
        "rounds": good.get("rounds"),
        "tip": good.get("tip"),
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
