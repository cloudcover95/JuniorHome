"""Months offline. One hash per close. No sign. No socket."""
import hashlib
def offline(days):
    prev = "0" * 16
    last = prev
    for i in range(days):
        last = hashlib.sha256(f"{prev}|{i}".encode()).hexdigest()[:16]
        prev = last
    return {"days": days, "closes": days, "tip": last, "bytes": days * 16,
            "signed": 0, "socket": False}
