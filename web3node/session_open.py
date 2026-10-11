"""Session opens without a power reading."""
import hashlib
def session(user):
    return {"user": user, "session": hashlib.sha256(user.encode()).hexdigest()[:16],
            "power_gate": False, "frozen": False, "disk": False}
