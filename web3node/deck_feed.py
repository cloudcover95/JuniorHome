"""Channel map feed. Engine reads this. No device open."""
from __future__ import annotations

import json
from pathlib import Path

from deck_math import map_channels

OUT = Path.home() / ".juniorhome" / "os" / "deck_feed.json"


def feed() -> dict:
    mapped = map_channels()
    body = {
        "protocol": "goldend-osai-omega/1",
        "feed": "channel-map",
        "rule": mapped["rule"],
        "n": len(mapped["channels"]),
        "energy": mapped["energy"],
        "channels": mapped["channels"],
        "source": "name-vector",
        "adc": False,
        "live": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body
