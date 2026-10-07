"""AbsMean trit and integer dot per JuniorDeck channel."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "deck_math.json"
CHANNELS = (
    "pads", "pad_banks", "keys", "aftertouch", "pitch", "mod",
    "knobs", "encoder", "sustain", "foot", "expression",
    "line_in", "line_out", "headphone", "cv_gate", "midi_din",
    "usb_audio", "usb_midi", "display",
)
W = (1, 0, -1, 1, 0, -1, 1, 0)


def frame(name: str) -> list[float]:
    return [((ord(c) % 9) - 4) / 4.0 for c in name[:8]]


def absmean(xs: list[float]) -> tuple[float, list[int]]:
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return gamma, out


def run() -> dict:
    rows = []
    for name in CHANNELS:
        xs = frame(name)
        gamma, trits = absmean(xs)
        dot = sum(a * b for a, b in zip(trits, W))
        rows.append({"name": name, "gamma": round(gamma, 4), "zeros": trits.count(0), "dot": dot})
    body = {
        "protocol": "goldend-osai-omega/1",
        "op": "absmean-dot",
        "n": len(rows),
        "rows": rows,
        "switch": "mx-hotswap",
        "live": False,
        "model_pull": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
