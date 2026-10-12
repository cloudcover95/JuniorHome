"""Chain id is the genesis hash. Hello carries it. No listen."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_genesis import genesis
from juniors_node import status

OUT = Path.home() / ".juniorhome" / "os" / "juniors_id.json"


def hello() -> dict:
    g = genesis()
    node = status()
    return {
        "ok": g["ok"] and node["ok"],
        "kind": "hello",
        "chain_id": g["hash"],
        "tip": node["tip"],
        "height": node["height"],
        "bind": "127.0.0.1",
        "listen": False,
        "peers": 0,
        "socket": False,
        "mint": False,
        "network": False,
        "layer_one": False,
    }


def safeguard() -> dict:
    row = hello()
    row["ok"] = row["ok"] and row["chain_id"] == "d1fbcc50da6edcc2"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row
