"""Bench the chain record close link. Opt-in, AntiCon, one trit pack, base rate 0.05, pool held, mint false, socket false. No poll sets rate. No model. No UE."""
import sys
import time
from chain_record_close_link import chain_record_close_link
from base_rate import BASE


def chain_record_close_link_bench(rounds=3):
    t0 = time.perf_counter()
    rows = []
    for _ in range(rounds):
        row = chain_record_close_link()
        rows.append(row)
        if not row.get("ok"):
            return {"ok": False, "why": row.get("why", "close_link"), "mint": False, "socket": False}
    us = round((time.perf_counter() - t0) * 1e6, 1)
    mean_us = round(sum(r.get("us", 0) for r in rows) / max(rounds, 1), 1) if rows else 0
    tips = {r.get("tip") for r in rows}
    held_ok = all(
        r.get("held") is True
        and r.get("rate") == BASE
        and r.get("released") == 0
        and r.get("mint") is False
        and r.get("socket") is False
        and r.get("poll_set") is False
        and r.get("closed") is True
        and r.get("linked") is True
        and r.get("in_pack") is True
        for r in rows
    )
    good = (
        len(rows) == rounds
        and held_ok
        and all(r.get("n", 0) >= 3 for r in rows)
        and len(tips) == 1
    )
    return {
        "ok": good,
        "us": us,
        "mean_us": mean_us,
        "rounds": rounds,
        "n": rows[0].get("n") if rows else 0,
        "tip": rows[0].get("tip") if rows else None,
        "prev": rows[-1].get("prev") if rows else None,
        "juniors": rows[0].get("juniors") if rows else None,
        "rate": BASE,
        "held": held_ok,
        "released": 0,
        "mint": False,
        "socket": False,
        "poll_set": False,
        "closed": True,
        "linked": True,
    }


def guard():
    good = chain_record_close_link_bench(3)
    bad = chain_record_close_link(sessions=[("dev", ["gaia"]), ("", ["prompt"])])
    ok = (
        good["ok"]
        and good["rounds"] == 3
        and good["rate"] == 0.05
        and good["held"] is True
        and good["closed"] is True
        and good["linked"] is True
        and good["mint"] is False
        and good["socket"] is False
        and good["poll_set"] is False
        and bad.get("ok") is False
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "mean_us": good.get("mean_us"),
        "rounds": good.get("rounds"),
        "n": good.get("n"),
        "tip": good.get("tip"),
        "juniors": good.get("juniors"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "closed": True,
        "linked": True,
        "mint": False,
        "socket": False,
        "poll_set": False,
    }


if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
