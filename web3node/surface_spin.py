"""Spin a surface from the OS mesh ticket. Does not launch a bundler."""
from __future__ import annotations

import json
from pathlib import Path

from os_mesh import mesh
from tritquant import ticket

ROOT = Path.home() / ".juniorhome" / "surfaces"
KINDS = ("react-native", "web", "legacy")


def spin(kind: str = "web", task: str = "JuniorDeck", env: str = "t4") -> dict:
    if kind not in KINDS:
        return {"ok": False, "reason": "kind", "kinds": list(KINDS)}
    row = mesh(task, env)
    quant = ticket([float(ord(c) % 17) / 17.0 for c in kind], [0.2, -0.1, 0.4, 0.0])
    body = {
        "protocol": "goldend-osai-omega/1",
        "kind": kind,
        "sha3": row["sha3"],
        "pack5": quant["pack5"],
        "hops": row["hops"],
        "env": env,
        "bind": "127.0.0.1",
        "model_pull": False,
        "metro": False,
        "live": False,
    }
    dest = ROOT / kind
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ticket.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "path": str(dest / "ticket.json"), **body}


def spin_all(task: str = "JuniorDeck", env: str = "t4") -> list[dict]:
    return [spin(k, task, env) for k in KINDS]
