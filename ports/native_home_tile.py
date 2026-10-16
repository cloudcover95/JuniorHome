"""Dash tile: read-only view of ~/.juniorhome/os/native_home.json (from web3node/native_home.py)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

NATIVE_HOME = Path.home() / ".juniorhome" / "os" / "native_home.json"

def tile(path: Path = NATIVE_HOME) -> dict[str, Any]:
    base: dict[str, Any] = {"tile": "native_home", "path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "native_home": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
    except (OSError, ValueError) as e:
        return {**base, "status": "unreadable", "error": str(e), "native_home": None}
    keys = ("home_present", "os_dir", "native", "local")
    return {**base, "status": "ok", "native_home": {k: data.get(k) for k in keys}}

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
