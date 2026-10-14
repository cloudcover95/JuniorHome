"""Home ROM module. Local file only. No fetch. No bundled ROM."""
from __future__ import annotations

import json
from pathlib import Path

INBOX = Path.home() / ".juniorhome" / "rom" / "inbox"
OUT = Path.home() / ".juniorhome" / "os" / "rom_module.json"
EXTS = {".nes", ".sfc", ".smc", ".gb", ".gbc", ".gba", ".md", ".bin", ".iso", ".cue"}


def scan() -> dict:
    INBOX.mkdir(parents=True, exist_ok=True)
    files = [p.name for p in INBOX.iterdir() if p.suffix.lower() in EXTS and p.is_file()]
    body = {
        "ok": True,
        "inbox": str(INBOX),
        "n": len(files),
        "files": files[:16],
        "running": False,
        "fetch": False,
        "bundled": False,
        "backend": None,
        "libretro": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
