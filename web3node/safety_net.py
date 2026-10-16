"""Safety net. Validates invariants. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from team_report import metrics

def check() -> dict[str, Any]:
    met = metrics()
    s = met["safety"]
    ok = (
        s["bind"] == "127.0.0.1"
        and s["flash"] is False
        and s["model_pull"] is False
        and s["mint"] is False
        and met["ternary"]["rate"] == 0.05
    )
    body = {"ok": ok, "safety": s, "rate": met["ternary"]["rate"], "local": True}
    out = Path.home() / ".juniorhome" / "os" / "safety_net.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
