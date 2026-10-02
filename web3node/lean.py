"""Compute the suite scores. One write."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "lean.json"


def _energy(note: str) -> float:
    xs = [((ord(c) % 5) - 2) / 2.0 for c in note[:32]] or [0.0]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    zeros = 0
    for x in xs:
        q = round(x / gamma)
        q = 1 if q > 1 else (-1 if q < -1 else int(q))
        zeros += q == 0
    return round(1.0 - zeros / len(xs), 3)


def run(note: str = "JuniorOS") -> dict:
    body = {
        "protocol": "goldend-osai-omega/1",
        "tool": "lean",
        "note": note[:160],
        "trit_energy": _energy(note),
        "harvest": "solar",
        "writes": 1,
        "boot": False,
        "train": False,
        "measured": False,
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body
