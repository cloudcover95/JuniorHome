"""Envelope hash must be a valid trit block."""
import sys
from envelope_full import envelope
from validate import make, valid
def check(user="dev"):
    row = envelope(user, {"prompt": 4, "gaia": 1}, True, True, "node")
    if not row["ok"]:
        return {"ok": False, "why": row.get("why")}
    block = valid(make(row["hash"]))
    return {"ok": block["ok"], "hash": block.get("hash"), "mint": False}
if __name__ == "__main__":
    import json
    row = check()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
