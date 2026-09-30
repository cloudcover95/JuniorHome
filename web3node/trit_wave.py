"""T0 trit waveform ticket for the gamma band. No MCU."""
from __future__ import annotations

import hashlib, json, math
from pathlib import Path


def clip(x: float) -> int:
    if x > 0.33:
        return 1
    if x < -0.33:
        return -1
    return 0


def pack_wave(n: int = 48, drift: float = 0.15) -> dict:
    seq = []
    for i in range(n):
        s = math.sin(2 * math.pi * i / 16) + drift * math.sin(2 * math.pi * i / 7)
        seq.append(clip(s))
    raw = ",".join(str(t) for t in seq).encode()
    return {
        "n": n,
        "trits": seq,
        "gamma_drift": drift,
        "sha3": hashlib.sha3_256(raw).hexdigest()[:16],
        "trit_mcu": False,
    }


def write_ticket(path: Path | None = None) -> Path:
    dest = path or Path.home() / ".juniorhome" / "deck" / "trit_ticket.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(pack_wave(), indent=2) + "\n", encoding="utf-8")
    return dest


if __name__ == "__main__":
    p = write_ticket()
    print(p.read_text(), end="")
