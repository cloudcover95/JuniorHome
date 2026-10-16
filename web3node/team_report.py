"""Dual-team report. Checks, profit, kernel metrics → Obsidian + brain. Local only."""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from checks_balance import check
from profit_terraform import layer as profit_layer

RATE = 0.05

def metrics() -> dict[str, Any]:
    return {
        "kernel": {"native": "bitplane.c", "viable_at": "512x512", "not_viable_at": "1024x1024", "lut": False},
        "ternary": {"mem_vs_fp32": 16.0, "pack": "2-bit", "rate": RATE},
        "stack": {"t0": "n<512*512", "t1": "n<1024*1024", "t2": "native or fallback"},
        "safety": {"bind": "127.0.0.1", "flash": False, "model_pull": False, "mint": False},
    }

def report(team: str = "llc", note: str = "team report") -> dict[str, Any]:
    bal = check(note)
    prof = profit_layer()
    met = metrics()
    text = json.dumps({"team": team, "balance": bal, "profit": prof, "metrics": met}, sort_keys=True, default=str)
    close = hashlib.sha256(text.encode()).hexdigest()[:16]
    body = {
        "team": team,
        "ts": datetime.now(timezone.utc).isoformat(),
        "balance": bal,
        "profit": {"ok": prof.get("balance", {}).get("ok"), "rows": len(prof.get("rows") or [])},
        "metrics": met,
        "close": close,
        "rate": RATE,
    }
    vault = Path.home() / ".juniorhome" / "obsidian" / "findings"
    vault.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M") + f"_{team}.md"
    (vault / name).write_text(
        f"---\nlocal: true\nteam: {team}\nclose: {close}\n---\n\n# {team} report\n\n"
        f"Balance ok: {bal.get('ok')}\n"
        f"Profit rows: {body['profit']['rows']}\n"
        f"Kernel: {met['kernel']}\n"
        f"Ternary: {met['ternary']}\n"
        f"Stack: {met['stack']}\n"
        f"Safety: {met['safety']}\n",
        encoding="utf-8",
    )
    brain = Path.home() / ".juniorhome" / "brain"
    brain.mkdir(parents=True, exist_ok=True)
    (brain / "team.jsonl").open("a", encoding="utf-8").write(json.dumps(body) + "\n")
    return body

if __name__ == "__main__":
    import sys
    team = sys.argv[1] if len(sys.argv) > 1 else "llc"
    print(json.dumps(report(team), indent=2, default=str))
