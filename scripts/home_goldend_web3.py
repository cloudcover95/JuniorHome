"""Home summary that includes goldend web3. Local only."""
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
        "goldend_web3": latest(base / "goldend_web3.json"),
        "test": latest(base / "goldend_web3_test.json"),
        "web3": latest(base / "cycle_web3.json"),
        "local": True,
    }
    out = base / "home_goldend_web3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
