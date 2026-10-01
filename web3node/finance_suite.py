"""Manager posts. Observer reads. No order, no RPC."""
from __future__ import annotations

import json
from pathlib import Path

from ledger import post, tail
from surface_compute import compute

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "observer.jsonl"


def manage(note: str = "stock") -> dict:
    spun = compute("web", note)
    row = post("JuniorStock", note, spun.get("pack5", ""))
    return {"posted": row["sha3"], "order": False, "pack5": row["pack5"]}


def observe(n: int = 8) -> dict:
    rows = tail(n)
    body = {
        "protocol": "goldend-osai-omega/1",
        "n": len(rows),
        "ports": sorted({r.get("port") for r in rows}),
        "orders": 0,
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"n": body["n"]}) + "\n")
    return body
