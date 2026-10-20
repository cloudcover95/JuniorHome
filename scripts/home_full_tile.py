"""Home tile that includes the full path. Local only."""
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
        "full_path": latest_json(base / "goldend_full_path.json"),
        "terraform": latest_json(base / "osai_terraform.json"),
        "efficiency": latest_json(base / "ternary_efficiency.json"),
        "layers": latest_json(base / "goldend_layers.json"),
        "local": True,
    }
    out = base / "home_full_tile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
