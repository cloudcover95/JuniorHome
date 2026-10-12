"""Block message. Carries text. Does not listen."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_rule import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "juniors_block.json"


def announce(text: str, prev: str = "0" * 16) -> dict:
    block = make(text, prev)
    check = valid(block)
    return {
        "ok": check["ok"],
        "kind": "block",
        "prev": block["prev"],
        "hash": block["hash"],
        "body": block["body"],
        "text": text,
        "bind": "127.0.0.1",
        "listen": False,
        "peers": 0,
        "mint": False,
    }


def safeguard() -> dict:
    msg = announce("home|0.05|1")
    bad = dict(msg)
    bad["text"] = "home|9|1"
    body = {
        "ok": msg["ok"] and not valid(bad)["ok"],
        "hash": msg["hash"],
        "listen": False,
        "peers": 0,
        "socket": False,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
