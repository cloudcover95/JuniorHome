"""FrameForge trunk and the GDK class map. Kit is not installed."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "gaming_gdk.json"


def status() -> dict:
    body = {
        "protocol": "goldend-osai-omega/1",
        "ok": True,
        "own": {
            "frameforge": "cloudcover95/FrameForge",
            "frameforge2d": "cloudcover95/FrameForge2D",
            "omega": "obj",
            "osai": "note-pack",
        },
        "communities": ["blender", "unreal", "web", "godot", "unity"],
        "gdk": {
            "name": "Microsoft Game Development Kit",
            "grdk": "runtime, MSIXVC, saves, TCUI",
            "gxdk": "Xbox headers, D3D12.x, xb tools",
            "win32": "WINAPI_FAMILY_GAMES",
            "graphics": "Direct3D12",
            "audio": "XAudio",
            "arm64": "preview",
            "installed": False,
            "certified": False,
        },
        "ue5_launch": False,
        "webgpu_launch": False,
        "decomp_third_party": False,
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
