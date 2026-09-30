#!/usr/bin/env python3
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from deck_port import analog_ok, gate, log_note
from deck_usb import probe, digitizer_ok
usb = probe()
ok = analog_ok(0.0, 10_000.0)
staff = gate([ok] * 6)
print(json.dumps({
    "usb": usb,
    "analog_ok": ok,
    "flagstaff": staff,
    "digitizer": digitizer_ok(ok, staff, usb),
    "row": log_note("usb probe"),
}, indent=2))
