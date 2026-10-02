"""Keep the higher trit-energy note. History only. No weight update."""
from __future__ import annotations

import json
from pathlib import Path

from trit_mesh import integrity

HIST = Path.home() / ".juniorhome" / "gaia_mesh" / "goldend_fetch.jsonl"
BEST = Path.home() / ".juniorhome" / "os" / "goldend_best.json"


def fetch(note: str = "JuniorOSai") -> dict:
    row = integrity(note)
    prev = 0.0
    if BEST.exists():
        prev = float(json.loads(BEST.read_text(encoding="utf-8")).get("energy") or 0)
    kept = note if row["energy"] >= prev else json.loads(BEST.read_text(encoding="utf-8")).get("note")
    body = {
        "protocol": "goldend-osai-omega/1",
        "species": "Goldend",
        "hops": ["Goldend", "JuniorFetch", "JuniorLLM", "JuniorOSai"],
        "note": kept,
        "energy": max(row["energy"], prev),
        "svd_1024": False,
        "train": False,
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    HIST.parent.mkdir(parents=True, exist_ok=True)
    with HIST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"energy": row["energy"], "kept": kept == note}) + "\n")
    BEST.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
