"""Multi-run compare. Local JSONL. No cloud."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def load(path: str | None = None) -> list[dict[str, Any]]:
    p = Path(path) if path else Path.home() / ".juniorhome" / "gaia_mesh" / "tuner.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    pa = float(a.get("dyno", {}).get("power_hp", a.get("power_hp", 0)))
    pb = float(b.get("dyno", {}).get("power_hp", b.get("power_hp", 0)))
    return {"a_hp": pa, "b_hp": pb, "delta_hp": round(pb - pa, 4)}

if __name__ == "__main__":
    print(json.dumps(compare({"power_hp": 100}, {"power_hp": 105}), indent=2))
