"""Home rail that runs the full stack. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def rail(note: str = "home rail") -> dict:
    from ports.full_stack import stack
    result = stack(note)
    body = {
        "energy": result["energy"]["value"],
        "asserted": result["goldend"]["asserted"],
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "home_full_stack.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(rail(), indent=2))
