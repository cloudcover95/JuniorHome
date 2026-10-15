"""Vault receipt. Local only. No network. No flash."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def receipt(close: str, kind: str = "tuner") -> dict[str, Any]:
    return {"close": close, "kind": kind, "flash": False, "rate": 0.05}

def write(close: str, kind: str = "tuner") -> Path:
    out = Path.home() / ".juniorhome" / "vault" / "tuner"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{close}.json"
    p.write_text(json.dumps(receipt(close, kind), indent=2), encoding="utf-8")
    return p

if __name__ == "__main__":
    print(write("deadbeef"))
