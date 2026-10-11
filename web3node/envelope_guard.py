"""Envelope guard. Isolated and bot fail."""
import sys, time
from envelope_full import envelope
def guard():
    t0 = time.perf_counter()
    off = envelope("dev", {"prompt": 1}, True, False)
    bot = envelope("", {"prompt": 1}, True, True)
    on = envelope("dev", {"prompt": 4, "gaia": 1}, True, True, "node")
    ok = off["why"] == "isolated" and bot["why"] == "bot" and on["ok"] and on["signed_here"] is False and on["mint"] is False
    return {"ok": ok, "us": round((time.perf_counter() - t0) * 1e6, 1)}
if __name__ == "__main__":
    import json
    row = guard()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
