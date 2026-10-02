"""Record one active surface. Does not create the others."""
from __future__ import annotations

import json
from pathlib import Path

OS = Path.home() / ".juniorhome" / "os"
SURFACES = ("code", "python", "blender", "llm")


def pick(name: str) -> dict:
    if name not in SURFACES:
        return {"ok": False, "surfaces": list(SURFACES)}
    OS.mkdir(parents=True, exist_ok=True)
    body = {"ok": True, "active": name, "others": False, "launch": False, "bind": "127.0.0.1"}
    (OS / "active.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body
