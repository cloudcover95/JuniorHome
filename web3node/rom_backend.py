"""Sniff a local ROM. Probe a local backend. No fetch."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from rom_module import INBOX, EXTS

OUT = Path.home() / ".juniorhome" / "os" / "rom_backend.json"
BINS = ("retroarch", "sameboy", "mgba", "mednafen")


def sniff(path: Path) -> str:
    head = path.read_bytes()[:16]
    if head.startswith(b"NES\x1a"):
        return "nes"
    if head[:3] in (b"SMC",) or path.suffix.lower() in {".sfc", ".smc"}:
        return "snes"
    if path.suffix.lower() in {".gb", ".gbc"}:
        return "gameboy"
    if path.suffix.lower() == ".gba":
        return "gba"
    return "unknown"


def probe() -> dict:
    INBOX.mkdir(parents=True, exist_ok=True)
    files = [p for p in INBOX.iterdir() if p.suffix.lower() in EXTS and p.is_file()]
    found = [b for b in BINS if shutil.which(b)]
    kinds = {p.name: sniff(p) for p in files[:8]}
    body = {
        "ok": True,
        "n": len(files),
        "kinds": kinds,
        "backends": found,
        "running": False,
        "fetch": False,
        "bundled": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
