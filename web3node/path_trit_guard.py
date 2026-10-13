"""Guard path under trit pack. Base rate held. Opt-in, AntiCon. mint false. socket false."""
import sys
import time
from base_rate import BASE
from path import path
from validate import make, valid, pack


def check_path(actor, hits, opted_in=True):
    row = path(actor, hits, opted_in)
    if not row.get("ok"):
        return row
    text = "|".join(hits)
    body = pack(text)
    block = make(text)
    check = valid(block)
    if not check["ok"] or any(b not in (0, 1, 2) for b in body):
        return {"ok": False, "why": "trit"}
    if row.get("rate") != BASE or row.get("released") != 0:
        return {"ok": False, "why": "rate"}
    if row.get("mint") is not False or row.get("socket") is not False:
        return {"ok": False, "why": "flags"}
    return {
        "ok": True,
        "hash": row.get("hash"),
        "rate": row["rate"],
        "released": row["released"],
        "held": True,
        "mint": False,
        "socket": False,
        "trit": len(body),
    }


def bench(n=6):
    t0 = time.perf_counter()
    goods = []
    for i in range(n):
        actor = "dev" if i % 2 == 0 else "home"
        hits = ["gaia", "home"] if i % 2 == 0 else ["field"]
        row = check_path(actor, hits, True)
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
        "avg_trit": round(sum(g["trit"] for g in goods) / len(goods), 1),
    }


def guard():
    good = bench(4)
    bad_iso = check_path("dev", ["gaia"], False)
    bad_bot = check_path("", ["gaia"], True)
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
    }


if __name__ == "__main__":
    import json

    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
