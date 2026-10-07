"""Production session ticket. No vendor arranger."""
from __future__ import annotations

import json
from pathlib import Path

from deck_channels import pack

OUT = Path.home() / ".juniorhome" / "os" / "deck_session.json"


def session(tracks: int = 8) -> dict:
    channels = pack()
    n = 8 if tracks < 1 else min(tracks, 16)
    body = {
        "protocol": "goldend-osai-omega/1",
        "board": "JuniorDeck",
        "tracks": n,
        "rate_hz": 48000,
        "bits": 24,
        "channels": channels["channels"].__len__(),
        "zeros": channels["zeros"],
        "switch": "mx-hotswap",
        "arranger": False,
        "plugin_host": False,
        "live": False,
        "bind": "127.0.0.1",
    }
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
