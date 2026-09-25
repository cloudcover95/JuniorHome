"""Drop a job file locally. fire=False always from Home."""
from __future__ import annotations

import json
from pathlib import Path

from ports.lab_catalog import CATALOG

SPOOL = Path.home() / ".juniorhome" / "lab" / "spool"


def queue(device: str, name: str, body: str) -> dict:
    ids = {c["id"] for c in CATALOG}
    if device not in ids:
        return {"ok": False, "device": device}
    SPOOL.mkdir(parents=True, exist_ok=True)
    dest = SPOOL / f"{device}-{name}"
    dest.write_text(body, encoding="utf-8")
    return {
        "ok": True,
        "path": str(dest),
        "fire": False,
        "bind": "127.0.0.1",
        "note": "operator starts the machine; this only wrote a file",
    }


def list_spool() -> list:
    if not SPOOL.is_dir():
        return []
    return sorted(p.name for p in SPOOL.iterdir() if p.is_file())
