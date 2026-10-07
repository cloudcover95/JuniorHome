"""Trit pack each JuniorDeck channel. SBC class. No vendor switch."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "deck_channels.json"
CHANNELS = (
    "pads", "pad_banks", "keys", "aftertouch", "pitch", "mod",
    "knobs", "encoder", "sustain", "foot", "expression",
    "line_in", "line_out", "headphone", "cv_gate", "midi_din",
    "usb_audio", "usb_midi", "display",
)


def _trit(name: str) -> int:
    q = round(((ord(name[0]) % 5) - 2) / 2)
    return 1 if q > 1 else (-1 if q < -1 else int(q))


def pack() -> dict:
    rows = []
    for name in CHANNELS:
        t = _trit(name)
        rows.append({"name": name, "trit": t, "switch": "mx-hotswap", "live": False})
    body = {
        "protocol": "goldend-osai-omega/1",
        "board": "JuniorDeck",
        "host": "sbc",
        "watts_class": 12,
        "measured": False,
        "switch": "mx-hotswap",
        "vendor_switch": False,
        "channels": rows,
        "zeros": sum(1 for r in rows if r["trit"] == 0),
        "open": ["usb-audio-class", "usb-midi-class", "midi-din", "cv-gate"],
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
