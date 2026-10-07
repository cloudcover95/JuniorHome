"""MIDI note pack from a dropped .mid. No port open. No plugin host."""
from __future__ import annotations

import json
from pathlib import Path

INBOX = Path.home() / ".juniorhome" / "deck" / "inbox"
OUT = Path.home() / ".juniorhome" / "os" / "deck_midi.json"


def _notes(raw: bytes) -> list[int]:
    found = []
    for i in range(len(raw) - 2):
        if raw[i] & 0xF0 == 0x90 and raw[i + 2] > 0:
            found.append(raw[i + 1])
            if len(found) == 8:
                break
    return found


def pack() -> dict:
    mids = sorted(INBOX.glob("*.mid")) if INBOX.exists() else []
    notes = _notes(mids[0].read_bytes()) if mids else []
    xs = [(n - 60) / 60.0 for n in notes] or [0.0] * 8
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = []
    for x in xs:
        q = round(x / gamma)
        trits.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    body = {
        "protocol": "goldend-osai-omega/1",
        "channel": "midi_din",
        "file": mids[0].name if mids else None,
        "notes": notes,
        "trits": trits,
        "usb_midi_class": True,
        "device_open": False,
        "plugin_host": False,
        "live": bool(notes),
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
