"""Wallet. Address, cold by default. No distribution."""
import hashlib
def open_wallet(user):
    return {"user": user, "address": hashlib.sha256(f"juniors|{user}".encode()).hexdigest()[:16],
            "cold": True, "sweep": False, "distribution": False, "mint": False,
            "node": "same formula", "network": False}
