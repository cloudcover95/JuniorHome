"""Network envelope. No listen. No peers."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_ledger import bench, walk

OUT = Path.home() / ".juniorhome" / "os" / "juniors_net.json"
KINDS = ("hello", "block", "tip")


def envelope(kind: str, tip: str) -> dict:
    if kind not in KINDS:
        return {"ok": False, "why": "kind"}
    return {
        "ok": True,
        "kind": kind,
        "tip": tip,
        "bind": "127.0.0.1",
        "listen": False,
        "peers": 0,
        "mint": False,
    }


def safeguard() -> dict:
    walked = walk()
    measured = bench(90)
    hello = envelope("hello", walked.get("tip") or "0" * 16)
    body = {
        "ok": hello["ok"] and measured["ok"],
        "kinds": list(KINDS),
        "hello": hello,
        "bench": measured,
        "listen": False,
        "peers": 0,
        "bind": "127.0.0.1",
        "mint": False,
        "socket": False,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
