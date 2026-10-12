"""Local ledger. Append packs, walk the tip. No peers."""
from __future__ import annotations

import json
import time
from pathlib import Path

from chain_record import run
from validate import valid

OUT = Path.home() / ".juniorhome" / "os" / "juniors_ledger.jsonl"


def append(sessions) -> dict:
    row = run(sessions)
    if not row.get("ok"):
        return row
    block = {
        "tip": row["tip"],
        "n": row["n"],
        "juniors": row["juniors"],
        "released": 0,
        "mint": False,
        "socket": False,
        "peers": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(block) + "\n")
    return block


def walk() -> dict:
    if not OUT.exists():
        return {"ok": True, "n": 0, "tip": None}
    lines = [json.loads(x) for x in OUT.read_text().splitlines() if x]
    ok = all(not row["mint"] and not row["socket"] and row["peers"] == 0 for row in lines)
    return {"ok": ok, "n": len(lines), "tip": lines[-1]["tip"] if lines else None}


def bench() -> dict:
    t0 = time.perf_counter()
    check = valid({"prev": "0" * 16, "hash": "0" * 16, "body": bytes([1, 1, 1])})
    us = (time.perf_counter() - t0) * 1e6
    return {"tamper_denied": not check["ok"], "us": round(us, 1)}


def safeguard() -> dict:
    led = append([("dev", ["gaia"]), ("home", ["os"])])
    walked = walk()
    measured = bench()
    return {
        "ok": bool(led.get("tip")) and walked["ok"],
        "tip": walked["tip"],
        "blocks": walked["n"],
        "bench": measured,
        "mint": False,
        "socket": False,
        "peers": 0,
        "network": False,
        "layer_one": False,
    }
