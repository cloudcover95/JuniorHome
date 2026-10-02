"""Mesh integrity is a ticket count. Not a 1024 SVD."""
from __future__ import annotations

import json
from pathlib import Path

INDEX = Path.home() / ".juniorhome" / "digest" / "index.json"
MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "mesh_integrity.jsonl"
VAULT = Path.home() / ".juniorhome" / "vault" / "bitnetCloud" / "notes"


def check() -> dict:
    n = 0
    if INDEX.exists():
        n = int(json.loads(INDEX.read_text(encoding="utf-8")).get("n") or 0)
    notes = len(list(VAULT.glob("*.md"))) if VAULT.exists() else 0
    body = {
        "protocol": "goldend-osai-omega/1",
        "ok": n >= 8,
        "digest_n": n,
        "vault_n": notes,
        "svd_1024": False,
        "cores": ["JuniorStock", "JuniorOmega", "AGI_SDK", "web3node", "JuniorOSai"],
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ok": body["ok"], "n": n}) + "\n")
    return body
