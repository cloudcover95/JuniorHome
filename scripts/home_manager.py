"""Home script that runs the production manager. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def rail(note: str = "home manager") -> dict:
    from ports.production_manager import run
    result = run(note)
    body = {"keep": result["keep"], "synced": result["synced"], "goldend_keep": result["goldend_keep"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "home_manager.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(rail(), indent=2))
