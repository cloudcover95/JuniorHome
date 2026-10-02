"""JuniorDeck CLI. Same contract as Home: one verb, one file. Status writes nothing."""
from __future__ import annotations
import json
from pathlib import Path
OS = Path.home() / ".juniorhome" / "deck"
SURFACES = ("midi", "audio", "plate", "screen")
HOSTS = ("linux", "darwin", "windows", "sbc", "legacy")
def status():
    active = {}
    path = OS / "active.json"
    if path.exists():
        active = json.loads(path.read_text(encoding="utf-8"))
    return {"cmd": "deck-status", "active": active.get("active"),
            "surfaces": {n: (OS / f"surface_{n}.json").exists() for n in SURFACES},
            "hosts": {n: (OS / f"host_{n}.json").exists() for n in HOSTS},
            "live": False, "writes": 0}
def pick(name):
    if name not in HOSTS:
        return {"ok": False, "hosts": list(HOSTS), "writes": 0}
    OS.mkdir(parents=True, exist_ok=True)
    body = {"ok": True, "active": name, "live": False, "bind": "127.0.0.1", "writes": 1}
    (OS / "active.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body
def dispatch(argv):
    verb = argv[0] if argv else "help"
    if verb == "status":
        return status()
    if verb == "pick" and len(argv) > 1:
        return pick(argv[1])
    if verb in SURFACES or verb in HOSTS:
        OS.mkdir(parents=True, exist_ok=True)
        kind = "surface" if verb in SURFACES else "host"
        body = {"kind": kind, "name": verb, "live": False, "ardour": False,
                "model_pull": False, "writes": 1, "bind": "127.0.0.1"}
        (OS / f"{kind}_{verb}.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
        return body
    return {"cmd": "deck-help", "verbs": ["status", "pick", *SURFACES, *HOSTS], "live": False}
