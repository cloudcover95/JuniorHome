"""OSS device classes. Local spool, no fire."""
from __future__ import annotations

CATALOG = [
    {"id": "klipper-moonraker", "kind": "fdm", "port": 7125, "files": [".gcode"]},
    {"id": "octoprint", "kind": "fdm", "port": 5000, "files": [".gcode"]},
    {"id": "grbl-laser", "kind": "laser", "port": None, "files": [".nc", ".gcode"]},
    {"id": "lightburn-export", "kind": "laser", "port": None, "files": [".lbrn2"]},
    {"id": "cups-ipp", "kind": "paper", "port": 631, "files": [".pdf", ".ps"]},
    {"id": "pi-gpio", "kind": "sbc", "port": None, "files": []},
    {"id": "ros2-loopback", "kind": "robot", "port": None, "files": []},
    {"id": "shop-screen", "kind": "calc", "port": None, "files": [".json"]},
]
