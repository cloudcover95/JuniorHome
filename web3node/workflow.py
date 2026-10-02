"""One engine pass, recorded on each suite hop."""
from __future__ import annotations

import json
from pathlib import Path

from engine import run as engine_run

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "workflow.jsonl"
HOPS = (
    "JuniorHome", "JuniorOS", "JuniorOSai", "JuniorLLM", "JuniorOmega",
    "AGI_SDK", "web3node", "JuniorStock", "JuniorSOL", "JuniorFetch",
)


def run(note: str = "JuniorOS") -> dict:
    receipt = engine_run(note)
    MESH.parent.mkdir(parents=True, exist_ok=True)
    for hop in HOPS:
        with MESH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"hop": hop, "energy": receipt["trit_energy"]}) + "\n")
    return {
        "protocol": "goldend-osai-omega/1",
        "hops": list(HOPS),
        "energy": receipt["trit_energy"],
        "harvest": receipt["harvest"],
        "boot": False,
        "train": False,
        "model_pull": False,
        "bind": "127.0.0.1",
    }
