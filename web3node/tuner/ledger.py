"""Local ledger. Append-only JSONL. No network. No flash."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

RATE = 0.05

def append(kind: str, row: dict[str, Any], prev: str = "") -> dict[str, Any]:
    text = json.dumps(row, sort_keys=True)
    body = hashlib.sha256(text.encode()).hexdigest()[:16]
    close = hashlib.sha256(f"{prev}|{body}|{text}".encode()).hexdigest()[:16]
    line = {"kind": kind, "row": row, "i2s": body, "close": close, "prev": prev, "rate": RATE, "flash": False}
    mesh = Path.home() / ".juniorhome" / "gaia_mesh"
    mesh.mkdir(parents=True, exist_ok=True)
    (mesh / "tuner.jsonl").open("a", encoding="utf-8").write(json.dumps(line) + "\n")
    return line

if __name__ == "__main__":
    print(json.dumps(append("test", {"rpm": 4500}), indent=2))
