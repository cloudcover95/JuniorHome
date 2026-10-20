"""Final production report. Local only."""
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

def report() -> dict:
    base = Path.home() / ".juniorhome" / "os"
    body = {
        "sync_rail": latest(base / "home_sync_rail.json"),
        "goldend_sync": latest(base / "goldend_sync.json"),
        "bench": latest(base / "sync_bench.json"),
        "local": True,
    }
    out = base / "final_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
