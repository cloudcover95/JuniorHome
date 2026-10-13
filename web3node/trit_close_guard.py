"""Trit close guard. Close under path: opt-in, AntiCon, one trit pack.
Base rate 0.05 held, pool held, mint false, socket false.
No poll sets the rate.
"""
import sys
import time
from base_rate import BASE
from path import path
from validate import make, valid, pack


def close_trit(actor, hits, opted_in=True):
    row = path(actor, hits, opted_in)
    if not row.get("ok"):
        return {"ok": False, "why": row.get("why"), "mint": False, "socket": False}
    text = "|".join(hits)
    body = pack(text)
    block = make(text, row.get("tip", "0" * 16))
    check = valid(block)
    if not check["ok"] or any(b not in (0, 1, 2) for b in body):
        return {"ok": False, "why": "trit", "mint": False, "socket": False}
    if row.get("rate") != BASE or row.get("released", 0) != 0:
        return {"ok": False, "why": "rate", "mint": False, "socket": False}
    if row.get("mint") is not False or row.get("socket") is not False:
        return {"ok": False, "why": "flags", "mint": False, "socket": False}
    return {
        "ok": True,
        "hash": check["hash"],
        "prev": block["prev"],
        "rate": row["rate"],
        "released": row["released"],
        "held": True,
        "juniors": row.get("juniors"),
        "mint": False,
        "socket": False,
        "trit": len(body),
        "in_pack": True,
    }


def bench(n=4):
    t0 = time.perf_counter()
    goods = []
    for i in range(n):
        actor = "dev" if i % 2 == 0 else "home"
        hits = ["gaia", "home"] if i % 2 == 0 else ["field", "deck"]
        row = close_trit(actor, hits, True)
        if not row.get("ok"):
            return {"ok": False, "why": row.get("why"), "i": i}
        goods.append(row)
    us = round((time.perf_counter() - t0) * 1e6, 1)
    return {
        "ok": True,
        "n": len(goods),
        "us": us,
        "rate": BASE,
        "released": 0,
        "held": True,
        "mint": False,
        "socket": False,
        "avg_trit": round(sum(g["trit"] for g in goods) / max(len(goods), 1), 1),
    }


def guard():
    good = bench(3)
    bad_iso = close_trit("dev", ["gaia"], False)
    bad_bot = close_trit("", ["gaia"], True)
    ok = (
        good["ok"]
        and good["rate"] == 0.05
        and good["held"] is True
        and good["mint"] is False
        and good["socket"] is False
        and bad_iso.get("ok") is False
        and bad_iso.get("why") == "isolated"
        and bad_bot.get("ok") is False
        and bad_bot.get("why") == "bot"
    )
    return {
        "ok": ok,
        "us": good.get("us"),
        "n": good.get("n"),
        "rate": good.get("rate"),
        "held": good.get("held"),
        "avg_trit": good.get("avg_trit"),
        "mint": False,
        "socket": False,
        "pool_held": True,
    }


if __name__ == "__main__":
    import json

    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
