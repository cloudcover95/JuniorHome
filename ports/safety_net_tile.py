"""Dash tile: read-only view of ~/.juniorhome/os/safety_net.json (from web3node/safety_net.py)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

SAFETY = Path.home() / ".juniorhome" / "os" / "safety_net.json"

def tile(path: Path = SAFETY) -> dict[str, Any]:
    base: dict[str, Any] = {"tile": "safety_net", "path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "safety_net": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
    except (OSError, ValueError) as e:
        return {**base, "status": "unreadable", "error": str(e), "safety_net": None}
    keys = ("ok", "safety", "rate", "local")
    return {**base, "status": "ok", "safety_net": {k: data.get(k) for k in keys}}

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
