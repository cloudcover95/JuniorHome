#!/usr/bin/env python3
"""Operator probe. Does not start OpenVPN, Slate, or a GL image."""
import json
from pathlib import Path

print(json.dumps({
    "tun0": Path("/sys/class/net/tun0").exists(),
    "openvpn_bin": Path("/usr/sbin/openvpn").exists(),
    "slate": False,
    "glinet": False,
    "bind": "127.0.0.1",
    "bind_all": False,
}, indent=2))
