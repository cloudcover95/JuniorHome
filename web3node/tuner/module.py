"""Home module registration. Opt-in. Local only. No flash."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

MODULE = {
    "id": "tuner",
    "title": "Tuner",
    "optional": True,
    "orient": ["landscape", "portrait"],
    "max": {"landscape": [1920, 1080], "portrait": [1080, 1920]},
    "flash": False,
    "rate": 0.05,
}

def register() -> Path:
    out = Path.home() / ".juniorhome" / "os" / "modules" / "tuner.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(MODULE, indent=2), encoding="utf-8")
    return out

def enabled() -> bool:
    p = Path.home() / ".juniorhome" / "os" / "modules" / "tuner.json"
    if not p.is_file():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("id") == "tuner"
    except json.JSONDecodeError:
        return False

if __name__ == "__main__":
    print(register())
