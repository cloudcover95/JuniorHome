"""Brain digest. Reads team.jsonl. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def latest(n: int = 5) -> list[dict[str, Any]]:
    p = Path.home() / ".juniorhome" / "brain" / "team.jsonl"
    if not p.is_file():
        return []
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-n:]

def write(n: int = 5) -> Path:
    out = Path.home() / ".juniorhome" / "os" / "brain_digest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": latest(n), "n": n}, indent=2, default=str), encoding="utf-8")
    return out

if __name__ == "__main__":
    print(write())
