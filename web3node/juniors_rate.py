"""Rate lives in the pack. A different rate is a different tip."""
from __future__ import annotations

import json
from pathlib import Path

from validate import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "juniors_rate.json"
BASE = 0.05


def close(rate: float) -> dict:
    block = make(f"home|{rate}|1")
    check = valid(block)
    return {"ok": check["ok"], "rate": rate, "hash": block["hash"], "released": 0}


def safeguard() -> dict:
    base = close(BASE)
    moved = close(0.06)
    body = {
        "ok": base["ok"] and moved["ok"] and base["hash"] != moved["hash"],
        "base_rate": BASE,
        "base_hash": base["hash"],
        "moved_hash": moved["hash"],
        "released": 0,
        "mint": False,
        "socket": False,
        "peers": 0,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
