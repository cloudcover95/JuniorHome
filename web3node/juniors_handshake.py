"""Handshake check. Secret key stays in memory. No socket."""
from __future__ import annotations

import json
import time
from pathlib import Path

OUT = Path.home() / ".juniorhome" / "os" / "juniors_handshake.json"


def safeguard() -> dict:
    try:
        from handshake import accept, enroll, hello
    except Exception as exc:
        body = {"ok": False, "why": "import", "detail": type(exc).__name__,
                "pq_ready": False, "socket": False, "layer_one": False}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        return body
    t0 = time.perf_counter()
    row = enroll("dev")
    hel = hello(row, "0" * 16)
    us = (time.perf_counter() - t0) * 1e6
    good = accept({row["name"]: row["pk"]}, hel)
    missing = accept({}, hel)
    body = {
        "ok": good["ok"] and not missing["ok"],
        "why_missing": missing["why"],
        "sign_us": round(us, 1),
        "pq_ready": True,
        "key_on_disk": False,
        "socket": False,
        "peers": 0,
        "listen": False,
        "mint": False,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
