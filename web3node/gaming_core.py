"""Home gaming core. Surfaces named. None launched."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "gaming_core.json"
SURFACES = ("wasm", "webgpu", "unity", "godot", "custom", "omega", "osai")


def status() -> dict:
    body = {
        "protocol": "goldend-osai-omega/1",
        "ok": True,
        "surfaces": list(SURFACES),
        "omega": "obj",
        "osai": "note-pack",
        "frameforge": "own-title",
        "apple_silicon": "class",
        "decomp_third_party": False,
        "wasm_launch": False,
        "webgpu_launch": False,
        "unity_launch": False,
        "godot_launch": False,
        "ue5_launch": False,
        "live": False,
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
