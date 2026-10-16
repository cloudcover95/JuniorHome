"""Dash tile: read-only view of ~/.juniorhome/os/deps.json (from web3node/deps.py)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

DEPS = Path.home() / ".juniorhome" / "os" / "deps.json"

def tile(path: Path = DEPS) -> dict[str, Any]:
    base: dict[str, Any] = {"tile": "deps", "path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "deps": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
    except (OSError, ValueError) as e:
        return {**base, "status": "unreadable", "error": str(e), "deps": None}
    mods = data.get("modules")
    mods = mods if isinstance(mods, dict) else {}
    present = sum(1 for v in mods.values() if v is True)
    return {**base, "status": "ok", "deps": {
        "modules": mods, "local": data.get("local"),
        "present": present, "total": len(mods),
        "absent": sorted(k for k, v in mods.items() if v is not True)}}

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
