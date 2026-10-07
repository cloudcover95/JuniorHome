"""JuniorDeck engine. Feed, then bus, then session."""
from __future__ import annotations

import json
from pathlib import Path

from deck_bus import mix
from deck_feed import feed
from deck_session import session

OUT = Path.home() / ".juniorhome" / "os" / "deck_engine.json"


def run(tracks: int = 8) -> dict:
    rows = feed()
    bus = mix()
    sess = session(tracks)
    body = {
        "protocol": "goldend-osai-omega/1",
        "engine": "JuniorDeck",
        "feed": rows["feed"],
        "n": rows["n"],
        "energy": rows["energy"],
        "bus": bus["bus"],
        "tracks": sess["tracks"],
        "rate_hz": 48000,
        "bits": 24,
        "adc": False,
        "plugin_host": False,
        "live": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
