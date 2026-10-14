"""Roster packed as an OSai note. No launch."""
from __future__ import annotations

import json
from pathlib import Path

from frameforge_actors import ROSTER
from juniors_rule import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "frameforge_osai.json"


def note() -> dict:
    text = "|".join(a["id"] for a in ROSTER) + "|0.05|1"
    block = make(text)
    check = valid(block)
    body = {
        "ok": check["ok"],
        "text": text,
        "hash": block["hash"],
        "n": len(ROSTER),
        "osai": "note-pack",
        "ue5_launch": False,
        "layer_one": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
