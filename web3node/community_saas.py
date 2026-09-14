"""Hash-linked community ledger. No Google. No Atlas."""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path

def token(tenant):
    return hashlib.sha256(f"junioros|{tenant}".encode()).hexdigest()[:16]

def credit(tenant, delta, note=""):
    path = Path(__file__).resolve().parent / "vault" / "ledger.jsonl"
    prev = "0" * 16
    if path.is_file():
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            prev = json.loads(lines[-1]).get("hash", prev)
    row = {"t": int(time.time()*1000), "tenant": tenant, "delta": delta, "note": note,
           "prev": prev, "google": False, "mongo": False}
    row["hash"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(row)+"\n")
    return row
