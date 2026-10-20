"""Seed local business crypto addresses. No chain interaction."""
import json
from pathlib import Path

addrs = {"local_a": "local-receipt-only-a", "local_b": "local-receipt-only-b"}
path = Path.home() / ".juniorhome" / "juniorstack" / "crypto_addresses.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(addrs, indent=2), encoding="utf-8")
print(path)
