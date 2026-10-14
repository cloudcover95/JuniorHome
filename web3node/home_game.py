"""One game tick. FrameForge2D is active. Nothing launches."""
from __future__ import annotations

import json
from pathlib import Path

from frameforge_port import status as forge
from gaming_core import status as core

OUT = Path.home() / ".juniorhome" / "os" / "home_game.json"


def tick() -> dict:
    port = forge()
    surfaces = core()
    body = {
        "ok": port["ok"] and surfaces["ok"],
        "active": port["active"],
        "web": port["web"],
        "unreal": port["unreal"],
        "surfaces": surfaces["surfaces"],
        "omega": "obj",
        "osai": "note-pack",
        "ue5_launch": False,
        "webgpu_launch": False,
        "gdk_installed": False,
        "decomp_third_party": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
