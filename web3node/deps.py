"""Home suite local deps. Reports what is present. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

MODULES = [
    "web3node.team_report",
    "web3node.prod_manager",
    "web3node.safety_net",
    "web3node.native_home",
    "web3node.brain_digest",
    "web3node.checks_balance",
    "web3node.profit_terraform",
    "ports.prod_manager_tile",
    "ports.safety_net_tile",
    "ports.native_home_tile",
]

def present() -> dict[str, Any]:
    found = {}
    for name in MODULES:
        try:
            __import__(name)
            found[name] = True
        except Exception:
            found[name] = False
    body = {"modules": found, "local": True}
    out = Path.home() / ".juniorhome" / "os" / "deps.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(present(), indent=2))
