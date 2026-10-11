"""Wallet guards. Over the pool or an exported key fails."""
import sys
from wallet import wallet
from yield_pool import yield_of
def guard():
    over = yield_of("dev", 50, [0.1])
    cold = wallet("dev", 5, cold=True)
    ok = over["ok"] is False and over["why"] == "over_pool" and cold["key_exported"] is False and cold["mint"] is False
    return {"ok": ok, "over_pool": over["why"], "key_exported": cold["key_exported"]}
if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
