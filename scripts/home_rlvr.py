"""Home suite workflow that reads RLVR logs and writes a status tile. Local only."""
from __future__ import annotations
import json
from pathlib import Path

RLVR = Path.home() / ".juniorhome" / "os" / "rlvr.jsonl"
TRIT = Path.home() / ".juniorhome" / "os" / "rlvr_trit.jsonl"
OUT = Path.home() / ".juniorhome" / "os" / "home_rlvr.json"

def latest(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None

def workflow() -> dict:
    body = {
        "rlvr": latest(RLVR),
        "trit": latest(TRIT),
        "local": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(workflow(), indent=2))
