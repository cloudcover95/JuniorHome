"""Ingest a community sample → trit pack → mesh. Local only. No flash."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .profiles import map_sample
from .channels import pack_sample

def ingest(profile: str, raw: dict[str, Any]) -> dict[str, Any]:
    mapped = map_sample(profile, raw)
    if not mapped.get("ok"):
        return mapped
    packed = pack_sample(mapped["sample"])
    mesh = Path.home() / ".juniorhome" / "gaia_mesh"
    mesh.mkdir(parents=True, exist_ok=True)
    line = {"kind": "tuner_ingest", "profile": profile, "packed": packed}
    (mesh / "tuner.jsonl").open("a", encoding="utf-8").write(json.dumps(line) + "\n")
    return {"ok": True, "close": packed["close"], "flash": False}

if __name__ == "__main__":
    print(json.dumps(ingest("speeduino", {"RPM": 4500, "AFR": 14.7, "CLT": 90}), indent=2))
