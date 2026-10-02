"""User-space Home session. Not a bootloader."""
from __future__ import annotations

import json
import time
from pathlib import Path

from digest import digest
from web3_mesh import join

OS = Path.home() / ".juniorhome" / "os"
TABLE = OS / "ps.jsonl"
SESSION = OS / "session.json"


def _log(cmd: str, ok: bool) -> None:
    OS.mkdir(parents=True, exist_ok=True)
    row = {"cmd": cmd, "ok": ok, "ts": int(time.time()), "pid": "userspace", "bind": "127.0.0.1"}
    with TABLE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def init() -> dict:
    OS.mkdir(parents=True, exist_ok=True)
    body = {"booted": False, "bind": "127.0.0.1", "ts": int(time.time()), "user": "local"}
    SESSION.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    _log("init", True)
    return body


def run(cmd: str) -> dict:
    verb = (cmd or "help").split()[0]
    if verb == "init":
        return init()
    if verb == "digest":
        out = digest("JuniorHome")
        _log(verb, True)
        return {"cmd": verb, "n": out["n"]}
    if verb == "mesh":
        out = join()
        _log(verb, True)
        return {"cmd": verb, "ledger_n": out["ledger_n"]}
    if verb == "up":
        init()
        d = digest("JuniorHome")
        m = join()
        _log("up", True)
        return {"cmd": "up", "digest": d["n"], "ledger_n": m["ledger_n"], "boot": False}
    if verb == "ps":
        n = len(TABLE.read_text(encoding="utf-8").strip().splitlines()) if TABLE.exists() else 0
        return {"cmd": verb, "n": n, "session": SESSION.exists(), "boot": False}
    _log("help", True)
    return {"cmd": "help", "verbs": ["init", "up", "digest", "mesh", "ps"], "boot": False}
