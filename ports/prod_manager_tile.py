"""Dash tile: read-only view of ~/.juniorhome/os/prod_manager.json (from web3node/prod_manager.py)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

PROD = Path.home() / ".juniorhome" / "os" / "prod_manager.json"

def tile(path: Path = PROD) -> dict[str, Any]:
    base: dict[str, Any] = {"tile": "prod_manager", "path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "prod": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
    except (OSError, ValueError) as e:
        return {**base, "status": "unreadable", "error": str(e), "prod": None}
    keys = ("team", "balance_ok", "profit_rows", "kernel_viable",
            "ternary_mem", "stack", "safety", "local")
    return {**base, "status": "ok", "prod": {k: data.get(k) for k in keys}}

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
