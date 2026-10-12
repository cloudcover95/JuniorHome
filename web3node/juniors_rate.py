"""Rate is committed beside the pack. Trit body can collide."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from validate import pack

OUT = Path.home() / ".juniorhome" / "os" / "juniors_rate.json"
BASE = 0.05


def close(rate: float) -> dict:
    body = pack(f"home|{rate}|1")
    digest = hashlib.sha256(("0" * 16).encode() + body + f"|{rate}".encode()).hexdigest()[:16]
    return {"ok": True, "rate": rate, "hash": digest, "released": 0}


def safeguard() -> dict:
    base = close(BASE)
    moved = close(0.06)
    body = {
        "ok": base["hash"] != moved["hash"],
        "base_rate": BASE,
        "base_hash": base["hash"],
        "moved_hash": moved["hash"],
        "trit_collision": True,
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
