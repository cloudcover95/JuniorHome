"""Record which suite cores received a note. No launch."""
from __future__ import annotations

import json
from pathlib import Path

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "suite_intervene.jsonl"
CORES = (
    "JuniorHome", "JuniorOS", "JuniorOSai", "JuniorLLM", "JuniorOmega",
    "web3node", "JuniorStock", "JuniorSOL", "JuniorPython-Suite",
)


def intervene(note: str = "JuniorOS") -> dict:
    body = {
        "protocol": "goldend-osai-omega/1",
        "note": note[:160],
        "cores": list(CORES),
        "llama": "header-only",
        "launch": False,
        "model_pull": False,
        "bind": "127.0.0.1",
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"n": len(CORES)}) + "\n")
    return body
