"""Default Home stack. A node hosts it. A module hit counts."""
import hashlib
from cloud_share import share
DEFAULT = ("gaia", "deck", "field", "koirig", "home")
def stack(extra=None):
    mods = list(DEFAULT)
    for name in extra or []:
        if name not in mods:
            mods.append(name)
    return {"env": "JuniorHome", "modules": mods, "personalized": bool(extra)}
def node_hit(module, extra=None):
    hosted = stack(extra)["modules"]
    if module not in hosted:
        return {"ok": False, "why": "not_hosted", "mint": False}
    cut = share({"user": 1, "node": 1})
    return {"ok": True, "module": module, "share": cut["rows"], "cloud": cut["cloud"],
            "receipt": hashlib.sha256(f"node|{module}".encode()).hexdigest()[:16],
            "mint": False, "gas": False, "network": False}
