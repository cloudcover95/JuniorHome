"""Home status aggregator for inference, layers, and goldend. Local only."""
from __future__ import annotations
import json
from pathlib import Path

FILES = [
    "inference_ledger.jsonl",
    "goldend_layers.json",
    "goldend_inference.json",
    "goldend_hillclimb.json",
    "osai_agents.json",
    "compute_smart.json",
]

def latest_jsonl(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None

def aggregate() -> dict:
    base = Path.home() / ".juniorhome" / "os"
    body = {"local": True}
    for name in FILES:
        p = base / name
        if name.endswith(".jsonl"):
            body[name] = latest_jsonl(p)
        elif p.exists():
            try:
                body[name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                body[name] = None
        else:
            body[name] = None
    out = base / "home_inference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(aggregate(), indent=2))
