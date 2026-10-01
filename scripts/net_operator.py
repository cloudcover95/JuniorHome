#!/usr/bin/env python3
"""Local net class probe. Does not start a tunnel."""
import json
from pathlib import Path

print(json.dumps({
    "tun0": Path("/sys/class/net/tun0").exists(),
    "bind": "127.0.0.1",
    "bind_all": False,
}, indent=2))
