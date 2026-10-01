"""Legacy host. Reads the spun ticket. No bundler."""
import json
from pathlib import Path

p = Path.home() / ".juniorhome" / "surfaces" / "legacy" / "ticket.json"
if not p.exists():
    print(json.dumps({"ok": False, "reason": "no ticket"}))
else:
    row = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps({"ok": row.get("bind") == "127.0.0.1", "sha3": row.get("sha3")}))
