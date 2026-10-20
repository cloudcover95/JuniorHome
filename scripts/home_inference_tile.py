"""Home status tile that aggregates inference, layers, and goldend. Local only."""
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
        "inference": latest_json(base / "inference_bench.json"),
        "layers": latest_json(base / "goldend_layers.json"),
        "hillclimb": latest_json(base / "goldend_hillclimb.json"),
        "agents": latest_json(base / "osai_agents.json"),
        "local": True,
    }
    out = base / "home_inference_tile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
