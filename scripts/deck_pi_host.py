#!/usr/bin/env python3
"""Pi / SFF host stub. Class-compliant USB later. live=False until binaries."""
import json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from trit_wave import pack_wave, write_ticket

hosts = {k: bool(shutil.which(k)) for k in ("ardour", "hydrogen", "audacity")}
ticket = write_ticket()
print(json.dumps({
    "live": False,
    "hosts": hosts,
    "ticket": str(ticket),
    "wave": pack_wave(),
    "usb_audio": False,
    "trit_mcu": False,
}, indent=2))
