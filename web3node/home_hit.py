"""Any JuniorHome action is a hit. Same formula."""
import hashlib
from cloud_share import share
ACTIONS = ("prompt", "dev", "agent")
def action(kind, name="home"):
    if kind not in ACTIONS:
        return {"ok": False, "why": "not_home", "mint": False}
    cut = share({"actor": 1, "home": 1})
    body = f"{kind}|{name}".encode()
    return {"ok": True, "env": "JuniorHome", "kind": kind, "share": cut["rows"],
            "cloud": cut["cloud"], "receipt": hashlib.sha256(body).hexdigest()[:16],
            "mint": False, "gas": False, "network": False}
