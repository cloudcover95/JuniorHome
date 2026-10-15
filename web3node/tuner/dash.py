"""Optional dash tile for tuner/dyno. Writes local JSON. No cloud."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def tile(row: dict[str, Any], orient: str = "landscape") -> dict[str, Any]:
    return {"module": "tuner", "orient": orient, "rpm": row.get("rpm"), "power_hp": row.get("power_hp"),
            "afr": row.get("afr"), "confirm": False}

def write(row: dict[str, Any], orient: str = "landscape") -> Path:
    out = Path.home() / ".juniorhome" / "os" / "tuner_dash.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tile(row, orient), indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    p = write({"rpm": 4500, "power_hp": 120, "afr": 14.7})
    print(p)
