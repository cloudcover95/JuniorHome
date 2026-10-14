"""One receipt. Audit and bot read this. No listen."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_node import status
from trit_link_bench import bench

OUT = Path.home() / ".juniorhome" / "os" / "juniors_receipt.json"


def receipt() -> dict:
    node = status()
    measured = bench(8)
    body = {
        "ok": node["ok"] and measured["ok"],
        "tip": node["tip"],
        "height": node["height"],
        "bench_tip": measured["tip"],
        "bench_us": measured["us"],
        "rate": 0.05,
        "supply": 0,
        "mint": False,
        "socket": False,
        "peers": 0,
        "listen": False,
        "layer_one": False,
        "actions": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
