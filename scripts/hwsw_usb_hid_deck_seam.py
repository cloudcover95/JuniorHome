#!/usr/bin/env python3
"""Hardware-to-software seam emulator: USB sound (ALSA / class 01) + USB HID (hidraw) -> JuniorDeck digitizer / audio inbox row.

What it emulates
  Presence of an emulated USB sound-class (01) device and/or USB HID hidraw
  node is gated with analog_ok(CV, Z) and the Flagstaff 6 AND, then turned
  into one JuniorDeck digitizer / audio_digest-style row. Trit pack is a short
  RMS-envelope byte string derived from an emulated HID report or PCM window
  (sample abs values mapped to trit codes 0/1/2, packed into bytes). Rows use
  the shared close rule sha256(prev || pack || text)[:16] with pack as real
  bytes, and carry the shared rate 0.05. digitizer may become true; live stays
  false.

What it does not touch
  No real /dev/snd, /dev/hidraw, or /dev/bus/usb. No sockets, no pyusb, no
  gadget configfs write, no Ardour/Hydrogen, no model or GGUF, no network.
  Nothing under ~/.juniorhome: inventory and rows live in a tempfile that is
  removed after the run. Standard library only. Flags: mint, socket,
  layer_one, fetch, ue5_launch all False; bind 127.0.0.1 is reported, never
  opened.

How to run
  python3 scripts/hwsw_usb_hid_deck_seam.py          # self-test, prints JSON
  python3 scripts/hwsw_usb_hid_deck_seam.py demo     # one emulated Deck row
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

RATE = 0.05
BIND = "127.0.0.1"
FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}
GENESIS = "0" * 16

# Flagstaff 6 AND — labels only (docs/DECK_USB.md mount order step 2).
FLAGSTAFF6 = (
    "cv_rail",
    "z_sense",
    "usb_c_cc",
    "deck_port",
    "fieldcore_ok",
    "inbox_ready",
)

# Emulated PCM / HID sample window (stdlib only; not real devices).
_EMU_PCM = bytes([12, 40, 80, 30, 5, 90, 55, 20, 70, 15, 45, 60])


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def trit_pack(samples: bytes) -> bytes:
    """Map abs sample values into trit codes 0/1/2, pack three trits per byte."""
    codes = []
    for s in samples:
        v = s if s < 128 else 256 - s
        if v < 22:
            codes.append(0)
        elif v < 64:
            codes.append(1)
        else:
            codes.append(2)
    out = bytearray()
    for i in range(0, len(codes), 3):
        chunk = codes[i : i + 3]
        while len(chunk) < 3:
            chunk.append(0)
        out.append(chunk[0] + 3 * chunk[1] + 9 * chunk[2])
    return bytes(out)


def inventory(devices: list[dict] | None = None) -> list[dict]:
    """In-memory / tempfile emulated USB class inventory."""
    if devices is not None:
        return list(devices)
    return [
        {"id": "alsa-usb-01", "kind": "usb_sound", "class": "01", "present": True},
        {"id": "hidraw-deck", "kind": "usb_hid", "class": "hidraw", "present": True},
    ]


def analog_ok(cv: float, z: float, cv_min: float = 0.4, z_max: float = 0.85) -> bool:
    return cv >= cv_min and 0.0 <= z <= z_max


def flagstaff6_and(gates: dict[str, bool]) -> bool:
    return all(gates.get(name, False) for name in FLAGSTAFF6)


def class_present(devs: list[dict]) -> bool:
    return any(
        d.get("present") and d.get("kind") in ("usb_sound", "usb_hid") for d in devs
    )


def digitizer_ready(
    cv: float,
    z: float,
    gates: dict[str, bool],
    devs: list[dict],
) -> bool:
    return analog_ok(cv, z) and flagstaff6_and(gates) and class_present(devs)


def build_row(
    prev: str,
    pack: bytes,
    text: str,
    digitizer: bool,
    lake: Path | None = None,
) -> dict:
    sha = close(prev, pack, text)
    row = {
        "node": "juniordeck",
        "kind": "audio_digest",
        "rate": RATE,
        "prev": prev,
        "close": sha,
        "fire": False,
        "bind": BIND,
        "digitizer": digitizer,
        "live": False,
        **FLAGS,
    }
    if lake is not None:
        lake.parent.mkdir(parents=True, exist_ok=True)
        with lake.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def default_gates(ok: bool = True) -> dict[str, bool]:
    return {name: ok for name in FLAGSTAFF6}


def self_test() -> dict:
    with tempfile.TemporaryDirectory() as td:
        lake = Path(td) / "deck_inbox" / "audio_digest.jsonl"
        pack = trit_pack(_EMU_PCM)
        assert isinstance(pack, bytes) and len(pack) > 0

        # Happy path: digitizer True, live False.
        gates = default_gates(True)
        devs = inventory()
        dig = digitizer_ready(0.7, 0.3, gates, devs)
        assert dig is True
        text = json.dumps(
            {"usb_sound": True, "usb_hid": True, "envelope": pack.hex()},
            sort_keys=True,
        )
        row = build_row(GENESIS, pack, text, dig, lake)
        assert row["digitizer"] is True and row["live"] is False
        assert row["rate"] == RATE and row["fire"] is False
        assert row["prev"] == GENESIS
        assert row["close"] == close(GENESIS, pack, text)
        assert not any(row[k] for k in FLAGS)
        assert row["bind"] == BIND

        # Missing class device -> digitizer False.
        missing = [
            {"id": "alsa-usb-01", "kind": "usb_sound", "class": "01", "present": False},
            {"id": "hidraw-deck", "kind": "usb_hid", "class": "hidraw", "present": False},
        ]
        assert digitizer_ready(0.7, 0.3, gates, missing) is False

        # Broken Flagstaff bit -> digitizer False.
        broken = default_gates(True)
        broken["fieldcore_ok"] = False
        assert digitizer_ready(0.7, 0.3, broken, devs) is False

        # analog_ok fail -> digitizer False.
        assert digitizer_ready(0.1, 0.3, gates, devs) is False
        assert digitizer_ready(0.7, 0.99, gates, devs) is False

        # close() verifies with trit bytes pack (not a JSON label string).
        assert close(GENESIS, pack, text) == row["close"]
        wrong = close(GENESIS, b"not-trit", text)
        assert wrong != row["close"]

        assert lake.is_file()
        lines = lake.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    return {
        "ok": True,
        "digitizer": True,
        "live": False,
        "rate": RATE,
        "bind": BIND,
        "pack_len": len(pack),
        **FLAGS,
        "fire": False,
    }


def demo() -> dict:
    with tempfile.TemporaryDirectory() as td:
        lake = Path(td) / "deck_inbox" / "audio_digest.jsonl"
        pack = trit_pack(_EMU_PCM)
        gates = default_gates(True)
        devs = inventory()
        dig = digitizer_ready(0.65, 0.4, gates, devs)
        text = json.dumps(
            {
                "devices": [{"id": d["id"], "kind": d["kind"], "present": d["present"]} for d in devs],
                "envelope": pack.hex(),
            },
            sort_keys=True,
        )
        row = build_row(GENESIS, pack, text, dig, lake)
        return {"row": row, "pack_hex": pack.hex(), "lake": str(lake)}


if __name__ == "__main__":
    out = demo() if (len(sys.argv) > 1 and sys.argv[1] == "demo") else self_test()
    print(json.dumps(out, indent=2))
