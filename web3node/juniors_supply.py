"""Supply stays 0. A mint close is denied."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_genesis import genesis

OUT = Path.home() / ".juniorhome" / "os" / "juniors_supply.json"


def close(mint: bool) -> dict:
    if mint:
        return {"ok": False, "why": "mint", "supply": 0}
    g = genesis()
    return {"ok": g["ok"], "supply": 0, "held": True, "hash": g["hash"]}


def safeguard() -> dict:
    good = close(False)
    bad = close(True)
    body = {
        "ok": good["ok"] and not bad["ok"],
        "supply": 0,
        "held": True,
        "genesis": good.get("hash"),
        "why_mint": bad.get("why"),
        "mint": False,
        "socket": False,
        "peers": 0,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
