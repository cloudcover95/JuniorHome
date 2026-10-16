"""Dash tile: read-only view of ~/.juniorhome/os/brain_digest.json (from web3node/brain_digest.py)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

DIGEST = Path.home() / ".juniorhome" / "os" / "brain_digest.json"

def tile(path: Path = DIGEST) -> dict[str, Any]:
    base: dict[str, Any] = {"tile": "brain_digest", "path": str(path)}
    if not path.is_file():
        return {**base, "status": "missing", "digest": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
    except (OSError, ValueError) as e:
        return {**base, "status": "unreadable", "error": str(e), "digest": None}
    keys = ("rows", "n")
    return {**base, "status": "ok", "digest": {k: data.get(k) for k in keys}}

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
