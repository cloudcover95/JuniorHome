"""Mesh integrity as trit energy. Not a 1024 SVD."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "trit_mesh.json"


def integrity(note: str = "JuniorHome") -> dict:
    xs = [((ord(c) % 5) - 2) / 2.0 for c in note[:32]] or [0.0]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = []
    for x in xs:
        q = round(x / gamma)
        trits.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    zeros = trits.count(0)
    body = {
        "protocol": "goldend-osai-omega/1",
        "n": len(trits),
        "zeros": zeros,
        "energy": round(1.0 - zeros / len(trits), 3),
        "svd_1024": False,
        "ports": ["JuniorStock", "JuniorOmega", "AGI_SDK", "vault", "web3node", "JuniorOSai"],
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
