"""Dash tile: read-only view of ~/.juniorhome/os/tuner_stack.json (from web3node/tuner/stack_hook.py)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

STACK = Path.home() / ".juniorhome" / "os" / "tuner_stack.json"

def tile(path: Path = STACK) -> dict[str, Any]:
    base: dict[str, Any] = {"tile": "tuner_stack", "path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "stack": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
    except (OSError, ValueError) as e:
        return {**base, "status": "unreadable", "error": str(e), "stack": None}
    keys = ("m", "k", "layer", "bench", "lut")
    return {**base, "status": "ok", "stack": {k: data.get(k) for k in keys}}

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
