"""Accept a block message only if prev is the tip."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_block import announce
from juniors_ledger import append, walk
from juniors_rule import valid

OUT = Path.home() / ".juniorhome" / "os" / "juniors_accept.json"


def accept(msg: dict) -> dict:
    if not valid(msg).get("ok"):
        return {"ok": False, "why": "hash"}
    tip = walk().get("tip") or "0" * 16
    if msg["prev"] != tip and tip != "0" * 16:
        return {"ok": False, "why": "prev", "tip": tip}
    return append(msg["text"], msg["prev"])


def safeguard() -> dict:
    first = append("home|0.05|1")
    msg = announce("home|0.06|1", first["hash"])
    good = accept(msg)
    bad = accept(announce("home|9|1", "f" * 16))
    body = {
        "ok": good.get("ok") and not bad.get("ok"),
        "tip": walk().get("tip"),
        "why_bad": bad.get("why"),
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
