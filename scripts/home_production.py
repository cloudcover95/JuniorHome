"""Home rail that runs the goldend retry and writes production status. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def rail(note: str = "production") -> dict:
    from ports.goldend_retry import run
    from ports.production_status import status
    retry = run(note)
    prod = status()
    body = {"retry_ok": retry["retry"]["ok"], "status": prod, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "home_production.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(rail(), indent=2))
