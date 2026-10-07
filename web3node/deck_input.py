"""Line-in pack from a dropped WAV. No device open."""
from __future__ import annotations

import json
import wave
from pathlib import Path

from deck_math import absmean, winsor

INBOX = Path.home() / ".juniorhome" / "deck" / "inbox"
OUT = Path.home() / ".juniorhome" / "os" / "deck_input.json"


def _pcm(path: Path) -> list[float]:
    with wave.open(str(path), "rb") as wf:
        n = min(wf.getnframes(), 4096)
        raw = wf.readframes(n)
        width = wf.getsampwidth()
    if width != 2 or not raw:
        return []
    xs = []
    for i in range(0, len(raw) - 1, 2):
        v = int.from_bytes(raw[i:i + 2], "little", signed=True)
        xs.append(v / 32768.0)
    return xs[:8]


def sample() -> dict:
    wavs = sorted(INBOX.glob("*.wav")) if INBOX.exists() else []
    xs = _pcm(wavs[0]) if wavs else []
    live = bool(xs)
    if not xs:
        xs = [0.0] * 8
    trits, gamma = absmean(winsor(xs))
    body = {
        "protocol": "goldend-osai-omega/1",
        "channel": "line_in",
        "file": wavs[0].name if wavs else None,
        "gamma": round(gamma, 4),
        "trits": trits,
        "rate_class": 48000,
        "bits_class": 24,
        "live": live,
        "device_open": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
