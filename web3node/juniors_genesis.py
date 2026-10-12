"""Fixed genesis. Same bytes every run."""
from __future__ import annotations

import json
from pathlib import Path

from validate import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "juniors_genesis.json"
TEXT = "juniors-genesis|0.05|0"


def genesis() -> dict:
    block = make(TEXT)
    check = valid(block)
    body = {
        "ok": check["ok"],
        "text": TEXT,
        "prev": block["prev"],
        "hash": block["hash"],
        "body": block["body"].hex(),
        "mint": False,
        "socket": False,
        "peers": 0,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
