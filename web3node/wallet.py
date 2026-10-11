"""Wallet. Juniors are the Cloud cut. No mint."""
import hashlib
from cloud_share import share
def address(user):
    return hashlib.sha256(f"juniors|{user}".encode()).hexdigest()[:16]
def wallet(user, hits, cold=True):
    cut = share({"actor": hits, "home": hits}) if hits else {"rows": {}, "cloud": 0}
    return {"user": user, "address": address(user), "juniors": cut.get("cloud", 0),
            "cold": cold, "key_exported": False, "sweep": False, "mint": False,
            "distribution": False, "network": False}
