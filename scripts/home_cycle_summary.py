"""Home status summary for the closed cycle. Local only."""
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
        "cycle": latest(base / "closed_cycle.json"),
        "bench": latest(base / "cycle_bench.json"),
        "goldend": latest(base / "goldend_cycle.json"),
        "handoff": latest(base / "osai_handoff.json"),
        "local": True,
    }
    out = base / "home_cycle_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
