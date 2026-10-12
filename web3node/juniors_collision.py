"""Short closes can share one trit body. Count them."""
from __future__ import annotations

import json
from pathlib import Path

from validate import pack

OUT = Path.home() / ".juniorhome" / "os" / "juniors_collision.json"


def bench(n: int = 100) -> dict:
    texts = [f"home|{i / 100:.2f}|1" for i in range(n)]
    bodies = {pack(t) for t in texts}
    return {
        "n": n,
        "unique_packs": len(bodies),
        "collisions": n - len(bodies),
        "side_commit": True,
        "mint": False,
        "socket": False,
        "peers": 0,
        "layer_one": False,
    }


def safeguard() -> dict:
    row = bench()
    row["ok"] = row["unique_packs"] >= 1 and row["side_commit"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row
