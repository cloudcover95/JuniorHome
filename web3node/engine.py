"""Home suite engine. One pass. No boot, no train, no wattmeter."""
from __future__ import annotations

import json
from pathlib import Path

from digest import digest
from goldend_fetch import fetch
from harvest import critique
from trit_mesh import integrity

OUT = Path.home() / ".juniorhome" / "os" / "engine.json"


def run(note: str = "JuniorOS") -> dict:
    d = digest(note)
    mesh = integrity(note)
    best = fetch(note)
    harvest = critique(note)
    body = {
        "protocol": "goldend-osai-omega/1",
        "tool": "junior-engine",
        "digest": d["n"],
        "trit_energy": mesh["energy"],
        "kept": best["note"],
        "harvest": harvest["best"],
        "boot": False,
        "train": False,
        "measured": False,
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
