"""JuniorDeck CLI. Status writes nothing. Control does not open a device."""
from __future__ import annotations
import json
from pathlib import Path
from deck_controls import control
OS = Path.home() / ".juniorhome" / "deck"
SURFACES = ("midi", "audio", "plate", "screen")
HOSTS = ("linux", "darwin", "windows", "sbc", "legacy")
def status():
    active = {}
    path = OS / "active.json"
    if path.exists():
        active = json.loads(path.read_text(encoding="utf-8"))
    return {"cmd": "deck-status", "active": active.get("active"),
            "live": False, "writes": 0, "opened": False}
def dispatch(argv):
    verb = argv[0] if argv else "help"
    if verb == "status":
        return status()
    if verb == "control":
        return control(argv[1] if len(argv) > 1 else "note")
    if verb == "pick" and len(argv) > 1 and argv[1] in HOSTS:
        OS.mkdir(parents=True, exist_ok=True)
        body = {"ok": True, "active": argv[1], "live": False, "writes": 1, "bind": "127.0.0.1"}
        (OS / "active.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
        return body
    if verb in SURFACES or verb in HOSTS:
        return {"kind": "surface" if verb in SURFACES else "host", "name": verb, "live": False, "writes": 0}
    return {"cmd": "deck-help", "verbs": ["status", "control", "pick", *SURFACES], "live": False}
