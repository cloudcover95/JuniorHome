"""Beta gate. Names what runs. Does not boot."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / ".juniorhome" / "os" / "beta.json"
CHECKS = {
    "cli": ROOT / "web3node" / "cli.py",
    "app": ROOT / "app" / "serve.py",
    "lean": ROOT / "web3node" / "lean.py",
    "shell": ROOT / "web3node" / "os_shell.py",
}


def gate() -> dict:
    present = {name: path.is_file() for name, path in CHECKS.items()}
    body = {
        "version": "0.9.0-beta",
        "present": present,
        "ok": all(present.values()),
        "boot": False,
        "store": False,
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
