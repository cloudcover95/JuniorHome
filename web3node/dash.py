"""Dash status for the local UI. Modules are optional."""
from __future__ import annotations

import json
from pathlib import Path

OS = Path.home() / ".juniorhome" / "os"
INDEX = Path.home() / ".juniorhome" / "digest" / "index.json"
OUT = OS / "dash.json"
BASE = ("home", "os", "osai", "llm")
OPTIONAL = ("omega", "deck", "web3", "stock", "sol", "vault", "stone", "engr", "forge", "asahi")


def status() -> dict:
    session = {}
    if (OS / "session.json").exists():
        session = json.loads((OS / "session.json").read_text(encoding="utf-8"))
    points = []
    if INDEX.exists():
        points = json.loads(INDEX.read_text(encoding="utf-8")).get("points") or []
    body = {
        "bind": "127.0.0.1",
        "boot": False,
        "watts_class": 45,
        "measured": False,
        "svd_1024": False,
        "base": [p for p in BASE if p in points] or list(BASE),
        "optional": [p for p in OPTIONAL if p in points],
        "session": bool(session),
        "model_pull": False,
    }
    OS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
