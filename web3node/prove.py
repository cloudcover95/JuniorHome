"""Recompute the chain. No authority."""
import hashlib
def prove(days, tip):
    prev = "0" * 16
    for i in range(days):
        prev = hashlib.sha256(f"{prev}|{i}".encode()).hexdigest()[:16]
    return {"ok": prev == tip, "tip": prev, "authority": False}
