"""Home tile that includes the web3 full path. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def latest_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def tile() -> dict:
    base = Path.home() / ".juniorhome" / "os"
    body = {
        "web3": latest_json(base / "goldend_web3.json"),
        "full_path": latest_json(base / "goldend_full_path.json"),
        "efficiency": latest_json(base / "efficiency_summary.json"),
        "terraform": latest_json(base / "osai_terraform.json"),
        "local": True,
    }
    out = base / "home_web3_tile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
