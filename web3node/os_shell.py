"""User-space Home session. Not a bootloader."""
from __future__ import annotations

import json
import time
from pathlib import Path

from digest import digest
from web3_mesh import join

TABLE = Path.home() / ".juniorhome" / "os" / "ps.jsonl"


def _log(cmd: str, ok: bool) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    row = {"cmd": cmd, "ok": ok, "ts": int(time.time()), "pid": "userspace", "bind": "127.0.0.1"}
    with TABLE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def run(cmd: str) -> dict:
    verb = (cmd or "help").split()[0]
    if verb == "digest":
        out = digest("JuniorHome")
        _log(verb, True)
        return {"cmd": verb, "n": out["n"]}
    if verb == "mesh":
        out = join()
        _log(verb, True)
        return {"cmd": verb, "ledger_n": out["ledger_n"]}
    if verb == "ps":
        n = 0
        if TABLE.exists():
            n = len(TABLE.read_text(encoding="utf-8").strip().splitlines())
        return {"cmd": verb, "n": n, "boot": False}
    _log("help", True)
    return {"cmd": "help", "verbs": ["digest", "mesh", "ps"], "boot": False, "bind": "127.0.0.1"}
