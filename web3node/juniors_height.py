"""Height 0 is genesis. Next close is height 1."""
from __future__ import annotations

import json
import time
from pathlib import Path

from juniors_genesis import genesis
from juniors_ledger import append, walk

OUT = Path.home() / ".juniorhome" / "os" / "juniors_height.json"


def run() -> dict:
    t0 = time.perf_counter()
    g = genesis()
    nxt = append("home|0.05|1", g["hash"])
    walked = walk()
    us = (time.perf_counter() - t0) * 1e6
    body = {
        "ok": g["ok"] and nxt["ok"] and walked["ok"],
        "genesis": g["hash"],
        "tip": walked["tip"],
        "height": walked["n"],
        "us": round(us, 1),
        "mint": False,
        "socket": False,
        "peers": 0,
        "listen": False,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
