"""Close rule. Hash is prev, pack, and the text."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from validate import pack

OUT = Path.home() / ".juniorhome" / "os" / "juniors_rule.json"


def make(text: str, prev: str = "0" * 16) -> dict:
    body = pack(text)
    digest = hashlib.sha256(prev.encode() + body + text.encode()).hexdigest()[:16]
    return {"prev": prev, "hash": digest, "body": body.hex(), "text": text}


def valid(row: dict) -> dict:
    body = bytes.fromhex(row["body"])
    if any(b not in (0, 1, 2) for b in body):
        return {"ok": False, "why": "not_trit"}
    digest = hashlib.sha256(row["prev"].encode() + body + row["text"].encode()).hexdigest()[:16]
    if digest != row["hash"]:
        return {"ok": False, "why": "hash"}
    return {"ok": True, "hash": digest}


def safeguard() -> dict:
    a = make("home|0.05|1")
    b = make("home|0.06|1")
    bad = dict(a)
    bad["text"] = "home|0.07|1"
    body = {
        "ok": valid(a)["ok"] and valid(b)["ok"] and a["hash"] != b["hash"] and not valid(bad)["ok"],
        "rule": "sha256(prev || pack || text)[:16]",
        "a": a["hash"],
        "b": b["hash"],
        "mint": False,
        "socket": False,
        "peers": 0,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
