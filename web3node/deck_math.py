"""AbsMean trit map for one deck channel. Stdlib."""
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


def vector(name: str, n: int = 8) -> list[float]:
    return [((ord(name[i % len(name)]) % 9) - 4) / 4.0 for i in range(n)]


def winsor(xs: list[float], p: float = 0.95) -> list[float]:
    ordered = sorted(abs(x) for x in xs)
    cap = ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))] or 1.0
    return [max(-cap, min(cap, x)) for x in xs]


def absmean(xs: list[float]) -> tuple[list[int], float]:
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / gamma)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return out, gamma


def pack5(trits: list[int]) -> str:
    acc = 0
    n = 0
    raw = bytearray()
    for t in trits:
        acc = acc * 3 + (t + 1)
        n += 1
        if n == 5:
            raw.append(acc)
            acc = 0
            n = 0
    if n:
        raw.append(acc)
    return raw.hex()


def map_channels() -> dict:
    rows = []
    for name in CHANNELS:
        xs = winsor(vector(name))
        trits, gamma = absmean(xs)
        rows.append({
            "name": name,
            "gamma": round(gamma, 4),
            "trits": trits,
            "zeros": trits.count(0),
            "pack5": pack5(trits),
            "switch": "mx-hotswap",
            "live": False,
        })
    body = {
        "protocol": "goldend-osai-omega/1",
        "rule": "winsor-p95-absmean",
        "channels": rows,
        "energy": round(1.0 - sum(r["zeros"] for r in rows) / (len(rows) * 8), 3),
        "measured": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
