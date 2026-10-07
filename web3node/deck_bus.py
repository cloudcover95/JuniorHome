"""Sum channel trits onto one bus. Integer add, clip."""
from __future__ import annotations

import json
from pathlib import Path

from deck_math import map_channels

OUT = Path.home() / ".juniorhome" / "os" / "deck_bus.json"


def mix() -> dict:
    rows = map_channels()["channels"]
    width = len(rows[0]["trits"])
    bus = [0] * width
    for row in rows:
        for i, t in enumerate(row["trits"]):
            bus[i] += t
    clipped = [1 if v > 1 else (-1 if v < -1 else v) for v in bus]
    body = {
        "protocol": "goldend-osai-omega/1",
        "bus": clipped,
        "pre_clip": bus,
        "tracks": 8,
        "rate_hz": 48000,
        "bits": 24,
        "plugin_host": False,
        "live": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
