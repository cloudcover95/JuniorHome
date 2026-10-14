"""FrameForge port. 2D is the web surface. Unreal stays off."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "frameforge_port.json"


def status() -> dict:
    body = {
        "protocol": "goldend-osai-omega/1",
        "ok": True,
        "unreal": "cloudcover95/FrameForge",
        "web": "cloudcover95/FrameForge2D",
        "active": "web",
        "sim_hz": 120,
        "snapshot_hz": 20,
        "roster": "original",
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
