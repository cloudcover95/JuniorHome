"""Node status. Reports the tip. Does not listen."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_legacy import split
from juniors_supply import close

OUT = Path.home() / ".juniorhome" / "os" / "juniors_node.json"


def status() -> dict:
    led = split()
    supply = close(False)
    body = {
        "ok": led["ok"] and supply["ok"],
        "name": "juniors",
        "tip": led["tip"],
        "height": led["live"],
        "legacy": led["legacy"],
        "supply": 0,
        "held": True,
        "bind": "127.0.0.1",
        "listen": False,
        "peers": 0,
        "socket": False,
        "mint": False,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
