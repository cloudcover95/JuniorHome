"""Status summary for the production cycle. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def latest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def summary() -> dict:
    base = Path.home() / ".juniorhome" / "os"
    body = {
        "cycle": latest(base / "production_cycle.json"),
        "gate": latest(base / "ternary_gate.json"),
        "home": latest(base / "home_production_cycle.json"),
        "local": True,
    }
    out = base / "production_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
