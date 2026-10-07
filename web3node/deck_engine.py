"""JuniorDeck engine. Math, bus, session. No device open."""
from __future__ import annotations

import json
from pathlib import Path

from deck_bus import mix
from deck_math import map_channels
from deck_session import session

OUT = Path.home() / ".juniorhome" / "os" / "deck_engine.json"


def run(tracks: int = 8) -> dict:
    math = map_channels()
    bus = mix()
    sess = session(tracks)
    body = {
        "protocol": "goldend-osai-omega/1",
        "engine": "JuniorDeck",
        "rule": math["rule"],
        "energy": math["energy"],
        "bus": bus["bus"],
        "tracks": sess["tracks"],
        "rate_hz": sess["rate_hz"],
        "bits": sess["bits"],
        "switch": "mx-hotswap",
        "plugin_host": False,
        "live": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
