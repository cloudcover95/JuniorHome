"""Join ledger, vault, stock, and SOL receipts. No RPC."""
from __future__ import annotations

import json
from pathlib import Path

from ledger import tail
from vault_mesh import write

MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "web3_mesh.jsonl"
STOCK = Path.home() / ".juniorhome" / "gaia_mesh" / "stock_surface.jsonl"
SOL = Path.home() / ".juniorhome" / "gaia_mesh" / "sol_surface.jsonl"


def _last(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def join() -> dict:
    vault = write()
    rows = tail(8)
    body = {
        "protocol": "goldend-osai-omega/1",
        "hops": ["JuniorHome", "web3node", "JuniorStock", "JuniorSOL", "JuniorOSai"],
        "ledger_n": len(rows),
        "vault_n": vault["n"],
        "stock": _last(STOCK).get("sha3"),
        "sol": _last(SOL).get("receipt"),
        "address": False,
        "rpc": False,
        "order": False,
        "bind": "127.0.0.1",
        "model_pull": False,
    }
    MESH.parent.mkdir(parents=True, exist_ok=True)
    with MESH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ledger_n": body["ledger_n"], "vault_n": body["vault_n"]}) + "\n")
    return body
