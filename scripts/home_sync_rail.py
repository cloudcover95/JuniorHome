"""Home rail that runs production with sync. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def rail(note: str = "home sync") -> dict:
    from ports.production_with_sync import run
    result = run(note)
    body = {"keep": result["cycle"]["keep"], "synced": result["sync"]["updated"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "home_sync_rail.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(rail(), indent=2))
