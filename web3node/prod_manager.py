"""Production manager check. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from team_report import metrics, report
from checks_balance import check
from profit_terraform import layer as profit_layer

def run(team: str = "llc") -> dict[str, Any]:
    bal = check("prod manager")
    prof = profit_layer()
    met = metrics()
    body = {
        "team": team,
        "balance_ok": bool(bal.get("ok")),
        "profit_rows": len(prof.get("rows") or []),
        "kernel_viable": met["kernel"]["viable_at"],
        "ternary_mem": met["ternary"]["mem_vs_fp32"],
        "stack": met["stack"],
        "safety": met["safety"],
        "local": True,
    }
    out = Path.home() / ".juniorhome" / "os" / "prod_manager.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    report(team)
    return body

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
