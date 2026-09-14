import json
from pathlib import Path
def verify():
    path = Path(__file__).resolve().parent / "vault" / "ledger.jsonl"
    if not path.is_file():
        return {"ok": True, "n": 0}
    prev, n = "0"*16, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if n and row.get("prev") != prev:
            return {"ok": False, "n": n}
        prev = row.get("hash", prev); n += 1
    return {"ok": True, "n": n, "tip": prev}
