"""Disk write only after a closed envelope. Open envelope stays in RAM."""
from __future__ import annotations
ENVELOPES = {"t4": {"files": 4, "chars": 256}, "host": {"files": 8, "chars": 1024}}
_RAM = {}
_COUNT = {"n": 0}
def allow(env, text):
    cap = ENVELOPES.get(env)
    if cap is None:
        return {"ok": False, "why": "open", "disk": False}
    if len(text) > cap["chars"]:
        return {"ok": False, "why": "over_chars", "disk": False}
    if _COUNT["n"] >= cap["files"]:
        return {"ok": False, "why": "over_files", "disk": False}
    return {"ok": True, "why": "closed", "disk": True}
def commit(env, path, text):
    row = allow(env, text)
    _RAM[str(path)] = text
    if not row["ok"]:
        row["ram"] = True
        return row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _COUNT["n"] += 1
    row["ram"] = False
    return row
