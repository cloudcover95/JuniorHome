"""Optional shop floor. StoneField-shaped jsonl. Loopback only."""
from __future__ import annotations

import json
from pathlib import Path

DIR = Path.home() / ".juniorhome" / "lab"
LEDGER = DIR / "floor.jsonl"
HTML = DIR / "companion.html"

SEEDS = [
    {"id": "printer-fdm", "kind": "printer", "net": "127.0.0.1"},
    {"id": "mill- benchtop", "kind": "mill", "net": "127.0.0.1"},
    {"id": "shop-screen", "kind": "calc", "net": "local"},
]


def spin(on: bool = True) -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    if on and not LEDGER.is_file():
        with LEDGER.open("w", encoding="utf-8") as f:
            for row in SEEDS:
                row = {**row, "id": row["id"].replace(" ", "-")}
                f.write(json.dumps(row) + "\n")
    HTML.write_text(
        "<!doctype html><meta charset=utf-8><title>lab companion</title>"
        "<p>optional companion — not a voice agent</p><pre>"
        + (LEDGER.read_text(encoding="utf-8") if LEDGER.is_file() else "")
        + "</pre>",
        encoding="utf-8",
    )
    n = sum(1 for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()) if LEDGER.is_file() else 0
    return {
        "on": on,
        "nodes": n,
        "jsonl": str(LEDGER),
        "companion": str(HTML),
        "bind": "127.0.0.1",
        "ue5": False,
        "cortana": False,
        "omega_launch": False,
    }
