"""Old rows stay. Only side-commit rows count."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_rule import valid

LEDGER = Path.home() / ".juniorhome" / "os" / "juniors_ledger.jsonl"
OUT = Path.home() / ".juniorhome" / "os" / "juniors_legacy.json"


def split() -> dict:
    if not LEDGER.exists():
        return {"ok": True, "live": 0, "legacy": 0, "tip": None}
    live = 0
    legacy = 0
    tip = None
    for line in LEDGER.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if "text" in row and valid(row)["ok"] and not row.get("mint") and not row.get("socket"):
            live += 1
            tip = row["hash"]
        else:
            legacy += 1
    return {"ok": True, "live": live, "legacy": legacy, "tip": tip, "deleted": 0}


def safeguard() -> dict:
    row = split()
    body = {
        **row,
        "mint": False,
        "socket": False,
        "peers": 0,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
