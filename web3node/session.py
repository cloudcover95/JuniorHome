"""One trit block per session. No power reading freezes it."""
import hashlib
def session(user, power):
    body = hashlib.sha256(user.encode()).hexdigest()[:16]
    if not power:
        return {"user": user, "session": body, "frozen": True, "why": "no_power",
                "sweep": False, "disk": False, "support": "hold"}
    return {"user": user, "session": body, "trit": 16, "frozen": False,
            "disk": False, "support": "open"}
