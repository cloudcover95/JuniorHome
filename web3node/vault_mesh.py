"""Write ledger rows as local vault notes. No sync service."""
from __future__ import annotations

import json
from pathlib import Path

from ledger import tail

VAULT = Path.home() / ".juniorhome" / "vault" / "bitnetCloud" / "notes"


def write(n: int = 8) -> dict:
    VAULT.mkdir(parents=True, exist_ok=True)
    rows = tail(n)
    paths = []
    for row in rows:
        sha = row.get("sha3") or "note"
        body = (
            "---\n"
            "local: true\n"
            "cloud: false\n"
            f"port: {row.get('port')}\n"
            f"sha3: {sha}\n"
            "order: false\n"
            "rpc: false\n"
            "---\n\n"
            f"# {row.get('port')} {sha}\n\n"
            f"{row.get('note', '')}\n\n"
            "#osai #web3node\n"
        )
        path = VAULT / f"{sha}.md"
        path.write_text(body, encoding="utf-8")
        paths.append(str(path))
    return {"n": len(paths), "vault": str(VAULT), "sync": False, "bind": "127.0.0.1"}
