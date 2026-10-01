"""OS mesh: Goldend + Geode + Field + LLM under one envelope. Caps, not a wattmeter."""
from __future__ import annotations
import hashlib, json, platform
from pathlib import Path
ENVELOPES = {
    "t4": {"watts": 12, "notes": 8, "chars": 256, "files": 4, "download_gb": 0.0},
    "host": {"watts": 45, "notes": 16, "chars": 1024, "files": 8, "download_gb": 0.0},
}
MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "os_mesh.jsonl"
DECK_CLASSES = {
    "linux": "jack-or-pipewire host", "darwin": "coreaudio host",
    "windows": "wasapi host", "sbc": "usb-hid ticket, no daw",
    "legacy": "class-compliant midi + pcm16 drop",
}
def fit(text, cap):
    t = text or ""
    return t if len(t) <= cap else t[: cap - 1] + "..."
def mesh(task="deck", env="t4"):
    key = env if env in ENVELOPES else "t4"
    budget = ENVELOPES[key]
    note = fit(task, budget["chars"])
    goldend = {"protocol": "goldend-osai-omega/1",
               "species": "JuniorDeck" if "deck" in note.lower() else "JuniorOS",
               "ue5_launch": False}
    geode = {"keys": {"rows": 8, "cols": 8, "switch": "MX-hotswap"},
             "display": {"inch": 10.1, "flat": True, "touch": True}, "launch": False}
    field = {"intent": note[:64], "bind": "127.0.0.1"}
    llm = {"port": goldend["species"], "download_gb": 0.0, "inference": "ticket"}
    hops = ["JuniorOSai", "Gaia", "Goldend", "Geode", "Field", "JuniorLLM"]
    body = {"protocol": goldend["protocol"], "hops": hops[: budget["notes"]],
            "goldend": goldend, "geode": geode, "field": field, "llm": llm,
            "deck": {"sys": platform.system().lower(),
                     "class": DECK_CLASSES.get(platform.system().lower(), DECK_CLASSES["legacy"]),
                     "compat": list(DECK_CLASSES)},
            "env": key, "watts": budget["watts"], "measured": False,
            "bind": "127.0.0.1", "model_pull": False,
            "sha3": hashlib.sha3_256(note.encode()).hexdigest()[:16]}
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sha3": body["sha3"], "env": key, "hops": body["hops"]}) + "\n")
    return body
