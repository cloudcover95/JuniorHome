"""Score code, blender, and llm. One write. No launch."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "terraform.json"


def _energy(note: str) -> float:
    xs = [((ord(c) % 5) - 2) / 2.0 for c in note[:32]] or [0.0]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    zeros = sum(1 for x in xs if abs(round(x / gamma)) == 0)
    return round(1.0 - zeros / len(xs), 3)


def run(note: str = "JuniorOSai") -> dict:
    surfaces = []
    for name in ("code", "blender", "llm"):
        surfaces.append({"name": name, "energy": _energy(note + name), "launch": False})
    best = max(surfaces, key=lambda s: s["energy"])
    body = {
        "protocol": "goldend-osai-omega/1",
        "hops": ["JuniorOSai", "Goldend"],
        "surfaces": surfaces,
        "pick": best["name"],
        "bpy": False,
        "model_pull": False,
        "writes": 1,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body
