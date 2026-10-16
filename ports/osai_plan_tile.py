"""Dash tile: read-only view of ~/.juniorhome/os/osai_plan.json (from web3node/osai_hook.py)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

PLAN = Path.home() / ".juniorhome" / "os" / "osai_plan.json"

def tile(path: Path = PLAN) -> dict[str, Any]:
    base: dict[str, Any] = {"tile": "osai_plan", "path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "plan": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {**base, "status": "unreadable", "error": str(e), "plan": None}
    keys = ("m", "k", "layer", "bench", "lut")
    return {**base, "status": "ok", "plan": {k: data.get(k) for k in keys}}

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
