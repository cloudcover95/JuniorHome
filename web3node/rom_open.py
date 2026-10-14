"""Open a local ROM. Dry-run unless --open and a core is on PATH."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from rom_backend import BINS, sniff
from rom_module import EXTS, INBOX

OUT = Path.home() / ".juniorhome" / "os" / "rom_open.json"


def pick() -> tuple[Path | None, str | None]:
    INBOX.mkdir(parents=True, exist_ok=True)
    files = [p for p in INBOX.iterdir() if p.suffix.lower() in EXTS and p.is_file()]
    core = next((b for b in BINS if shutil.which(b)), None)
    return (files[0] if files else None, core)


def open_rom(do_open: bool = False) -> dict:
    path, core = pick()
    if path is None:
        body = {"ok": False, "why": "no_rom", "fetch": False, "running": False}
    elif core is None:
        body = {"ok": False, "why": "no_core", "file": path.name, "kind": sniff(path),
                "fetch": False, "running": False}
    else:
        argv = [core, str(path)]
        ran = False
        if do_open:
            subprocess.Popen(argv)
            ran = True
        body = {"ok": True, "file": path.name, "kind": sniff(path), "core": core,
                "argv": argv, "running": ran, "fetch": False, "bundled": False}
    body["bind"] = "127.0.0.1"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
