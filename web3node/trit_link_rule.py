"""Two closes linked under sha256(prev || pack || text)."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_rule import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "trit_link_rule.json"


def link() -> dict:
    a = make("dev|gaia|0.05|1")
    b = make("home|prompt|0.05|1", a["hash"])
    check = valid(b)
    body = {
        "ok": valid(a)["ok"] and check["ok"] and a["hash"] != b["hash"],
        "a": a["hash"],
        "tip": b["hash"],
        "rate": 0.05,
        "released": 0,
        "mint": False,
        "socket": False,
        "peers": 0,
        "listen": False,
        "layer_one": False,
        "rule": "side-commit",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
