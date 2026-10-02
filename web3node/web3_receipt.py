"""Web3 receipt. Hex is not an address. No RPC."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ledger import post

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "web3_receipt.jsonl"


def receipt(note: str = "web3", pack5: str = "") -> dict:
    row = post("web3node", note, pack5)
    body = {
        "protocol": "goldend-osai-omega/1",
        "port": "web3node",
        "ledger": row["sha3"],
        "receipt": hashlib.sha3_256(pack5.encode() or note.encode()).hexdigest()[:16],
        "address": False,
        "rpc": False,
        "order": False,
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"receipt": body["receipt"], "ledger": body["ledger"]}) + "\n")
    return body
