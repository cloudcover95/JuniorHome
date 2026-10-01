"""Append-only local ledger. Not a broker."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

LEDGER = Path.home() / ".juniorhome" / "gaia_mesh" / "ledger.jsonl"


def post(port: str, note: str, pack5: str = "") -> dict:
    body = {
        "protocol": "goldend-osai-omega/1",
        "port": port,
        "note": note[:160],
        "pack5": pack5,
        "sha3": hashlib.sha3_256(f"{port}:{note}".encode()).hexdigest()[:16],
        "ts": int(time.time()),
        "order": False,
        "rpc": False,
        "bind": "127.0.0.1",
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(body) + "\n")
    return body


def tail(n: int = 8) -> list[dict]:
    if not LEDGER.exists():
        return []
    lines = LEDGER.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
