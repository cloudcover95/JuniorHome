"""Bounded JuniorOS tick. Runs agent cycle and refreshes Home status. Local only."""
from __future__ import annotations
import json
from pathlib import Path

def tick(note: str = "os tick") -> dict:
    from ports.junioros_agent import run
    from scripts.home_rlvr import workflow
    agent = run(note)
    status = workflow()
    body = {"agent": agent.get("envelope"), "status": status.get("trit"), "local": True}
    out = Path.home() / ".juniorhome" / "os" / "junioros_tick.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(tick(), indent=2))
