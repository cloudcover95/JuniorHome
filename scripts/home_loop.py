"""Home rail that runs the closed loop. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def rail(note: str = "home loop") -> dict:
    from ports.closed_loop import loop
    result = loop(note)
    body = {"energy_ok": result["energy_ok"], "agent_updated": result["agent_updated"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "home_loop.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(rail(), indent=2))
