"""Final status with goldend manager. Local only."""
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

def status() -> dict:
    base = Path.home() / ".juniorhome" / "os"
    body = {
        "manager_energy": latest(base / "manager_with_energy.json"),
        "goldend_manager": latest(base / "goldend_manager.json"),
        "energy_check": latest(base / "energy_check.json"),
        "local": True,
    }
    out = base / "final_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
