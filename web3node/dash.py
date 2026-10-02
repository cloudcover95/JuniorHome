"""Write the dash ticket from the session. No measured watts."""
from __future__ import annotations

import json
from pathlib import Path

OS = Path.home() / ".juniorhome" / "os"
DIGEST = Path.home() / ".juniorhome" / "digest" / "index.json"
OUT = Path.home() / ".juniorhome" / "surfaces" / "web" / "dash.json"


def write() -> dict:
    session = {}
    if (OS / "session.json").exists():
        session = json.loads((OS / "session.json").read_text(encoding="utf-8"))
    ps = 0
    if (OS / "ps.jsonl").exists():
        ps = len((OS / "ps.jsonl").read_text(encoding="utf-8").strip().splitlines())
    digest_n = 0
    if DIGEST.exists():
        digest_n = json.loads(DIGEST.read_text(encoding="utf-8")).get("n", 0)
    body = {
        "title": "JuniorHome",
        "bind": "127.0.0.1",
        "port": 8765,
        "connected": False,
        "boot": False,
        "session": bool(session),
        "ps": ps,
        "digest_n": digest_n,
        "envelope": "t4",
        "watts": 45,
        "measured": False,
        "svd": False,
        "vendor_node": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
