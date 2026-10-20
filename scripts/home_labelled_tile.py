"""Home tile that includes labelled assert. Local only."""
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

def tile() -> dict:
    base = Path.home() / ".juniorhome" / "os"
    body = {
        "labelled": latest(base / "goldend_labelled.json"),
        "assert": latest(base / "labelled_assert.json"),
        "full_path": latest(base / "goldend_full_path.json"),
        "local": True,
    }
    out = base / "home_labelled_tile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
