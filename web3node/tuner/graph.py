"""Graph data prep. Local JSON. No cloud."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .runs import load

def series(path: str | None = None) -> dict[str, Any]:
    rows = load(path)
    rpm, hp = [], []
    for r in rows:
        d = r.get("dyno", r)
        if "rpm" in d and "power_hp" in d:
            rpm.append(d["rpm"])
            hp.append(d["power_hp"])
    return {"rpm": rpm, "power_hp": hp, "n": len(rpm), "flash": False}

def write(path: str | None = None) -> Path:
    out = Path.home() / ".juniorhome" / "os" / "tuner_graph.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(series(path), indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    print(write())
