"""Local ledger. Each row keeps prev, pack, and hash."""
from __future__ import annotations

import json
import time
from pathlib import Path

from validate import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "juniors_ledger.jsonl"


def append(text: str, prev: str = "0" * 16) -> dict:
    block = make(text, prev)
    check = valid(block)
    if not check["ok"]:
        return check
    row = {
        "prev": block["prev"],
        "hash": block["hash"],
        "body": block["body"].hex(),
        "mint": False,
        "socket": False,
        "peers": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return {"ok": True, "hash": row["hash"], "prev": prev}


def walk() -> dict:
    if not OUT.exists():
        return {"ok": True, "n": 0, "tip": None}
    lines = [json.loads(x) for x in OUT.read_text().splitlines() if x]
    ok = True
    for row in lines:
        body = bytes.fromhex(row["body"])
        check = valid({"prev": row["prev"], "hash": row["hash"], "body": body})
        if not check["ok"] or row["mint"] or row["socket"] or row["peers"]:
            ok = False
            break
    return {"ok": ok, "n": len(lines), "tip": lines[-1]["hash"] if lines else None}


def bench(n: int = 90) -> dict:
    t0 = time.perf_counter()
    prev = "0" * 16
    for i in range(n):
        row = make(f"close-{i}|0.05|1", prev)
        if not valid(row)["ok"]:
            return {"ok": False, "n": i}
        prev = row["hash"]
    us = (time.perf_counter() - t0) * 1e6
    return {"ok": True, "n": n, "us": round(us, 1), "bytes": n * 16, "tip": prev}


def safeguard() -> dict:
    first = append("gaia|0.05|1")
    second = append("home|0.05|1", first["hash"])
    walked = walk()
    measured = bench()
    return {
        "ok": first["ok"] and second["ok"] and walked["ok"] and measured["ok"],
        "tip": walked["tip"],
        "blocks": walked["n"],
        "bench": measured,
        "mint": False,
        "socket": False,
        "peers": 0,
        "network": False,
        "layer_one": False,
    }
