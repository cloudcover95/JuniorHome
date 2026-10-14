"""Bench the side-commit link. Two texts, one tip."""
from __future__ import annotations

import json
import time
from pathlib import Path

from juniors_rule import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "trit_link_bench.json"


def bench(n: int = 90) -> dict:
    t0 = time.perf_counter()
    prev = "0" * 16
    for i in range(n):
        a = make(f"dev|{i}|0.05|1", prev)
        b = make(f"home|{i}|0.05|1", a["hash"])
        if not valid(a)["ok"] or not valid(b)["ok"] or a["hash"] == b["hash"]:
            return {"ok": False, "n": i}
        prev = b["hash"]
    us = (time.perf_counter() - t0) * 1e6
    return {
        "ok": True,
        "n": n,
        "pairs": n,
        "us": round(us, 1),
        "tip": prev,
        "rate": 0.05,
        "mint": False,
        "socket": False,
        "peers": 0,
        "layer_one": False,
    }


def run() -> dict:
    row = bench()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row
