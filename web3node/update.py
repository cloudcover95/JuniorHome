"""An update is a completed action. No PoW. No protocol change."""
import hashlib
from cloud_share import share
def update(user):
    cut = share({"actor": 1, "home": 1})
    return {"ok": True, "kind": "update", "user": user,
            "receipt": hashlib.sha256(f"update|{user}".encode()).hexdigest()[:16],
            "cloud": cut["cloud"], "pow": False, "protocol_changed": False, "network": False}
