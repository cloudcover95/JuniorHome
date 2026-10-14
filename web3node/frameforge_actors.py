"""Original roster as a feed. No launch."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "frameforge_actors.json"
ROSTER = (
    {"id": "owl", "role": "scout"},
    {"id": "ranger", "role": "ranger"},
    {"id": "mage", "role": "mage"},
    {"id": "automaton", "role": "crystal"},
)


def feed() -> dict:
    body = {
        "ok": True,
        "title": "FrameForge",
        "active": "web",
        "actors": list(ROSTER),
        "n": len(ROSTER),
        "omega": "obj",
        "osai": "note-pack",
        "ue5_launch": False,
        "webgpu_launch": False,
        "nintendo": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
