"""Join deck audio ticket and tritquant. Loopback. No model pull."""
from __future__ import annotations

import json
from pathlib import Path

from tritquant import ticket

DECK = Path.home() / ".juniorhome" / "deck"
OUT = DECK / "osai_join.json"
MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "osai_join.jsonl"


def _audio() -> dict:
    path = DECK / "audio_digest.json"
    if not path.exists():
        return {"n": 0, "rows": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"n": 0, "rows": []}


def join() -> dict:
    audio = _audio()
    wave = [0.2, -0.1, 0.4, 0.0, -0.3, 0.15, 0.05, -0.2]
    rows = audio.get("rows") or []
    if rows and isinstance(rows[0].get("wave"), dict):
        trits = rows[0]["wave"].get("trits") or []
        if trits:
            wave = [float(t) for t in trits]
    quant = ticket(wave, wave)
    body = {
        "protocol": "goldend-osai-omega/1",
        "module": "cad",
        "audio_n": audio.get("n", 0),
        "objectives": (rows[0].get("objectives") if rows else []) or [],
        "trit": {"sha3": quant["sha3"], "y": quant["y"], "zeros": quant["zeros"], "pack5": quant["pack5"]},
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"sha3": quant["sha3"], "audio_n": body["audio_n"]}) + "\n")
    return body
