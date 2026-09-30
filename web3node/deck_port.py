"""Deck cad port. Loopback only. I2_S is a note, not a KEM."""
from __future__ import annotations

import json, time
from pathlib import Path

from trit_wave import pack_wave

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "deck.jsonl"


def analog_ok(cv: float, z_ohm: float) -> bool:
    return -5.0 <= cv <= 5.0 and 1.0 <= z_ohm <= 100_000.0


def gate(votes: list[bool]) -> bool:
    return len(votes) == 6 and all(votes)


def log_note(note: str, cv: float = 0.0, z_ohm: float = 10_000.0) -> dict:
    t0 = time.perf_counter()
    wave = pack_wave()
    ok = analog_ok(cv, z_ohm)
    votes = [ok] * 6
    row = {
        "protocol": "goldend-osai-omega/1",
        "module": "cad",
        "note": note[:80],
        "analog_ok": ok,
        "flagstaff": gate(votes),
        "wave_sha3": wave["sha3"],
        "ms": round((time.perf_counter() - t0) * 1000.0, 4),
        "bind": "127.0.0.1",
        "trit_mcu": False,
        "ml_kem": False,
        "ue5_launch": False,
        "live": False,
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    print(json.dumps(log_note("deck port tick"), indent=2))
