"""USB-C / USB3 class-compliant probe. No gadget until /dev exists."""
from __future__ import annotations

import os
from pathlib import Path

AUDIO = ("/dev/snd/pcmC0D0p", "/dev/snd/pcmC0D0c", "/dev/snd/controlC0")
HID = ("/dev/hidraw0", "/dev/hidraw1")
USB = ("/dev/bus/usb",)


def _any(paths: tuple[str, ...]) -> bool:
    return any(Path(p).exists() for p in paths)


def probe() -> dict:
    audio = _any(AUDIO)
    hid = _any(HID)
    bus = Path("/dev/bus/usb").is_dir() and any(Path("/dev/bus/usb").rglob("*"))
    return {
        "usb_c": True,
        "usb3_ss": bus,
        "class_audio": audio,
        "class_hid": hid,
        "gadget": Path("/sys/kernel/config/usb_gadget").is_dir(),
        "present": audio or hid,
        "live": False,
    }


def digitizer_ok(analog: bool, flagstaff: bool, probe_row: dict | None = None) -> bool:
    row = probe_row or probe()
    return bool(analog and flagstaff and row.get("present"))
