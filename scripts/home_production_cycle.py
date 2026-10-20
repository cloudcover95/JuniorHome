"""Home rail that runs the production cycle. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def rail(note: str = "home production") -> dict:
    from ports.production_cycle import cycle
    result = cycle(note)
    body = {"keep": result["keep"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "home_production_cycle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(rail(), indent=2))
