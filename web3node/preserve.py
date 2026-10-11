"""A dropped session keeps counts and the share."""
from cloud_share import share
def drop(user, hits):
    total = sum(hits.values())
    cut = share({"actor": total, "home": total}) if total else {"rows": {}, "cloud": 0}
    return {"user": user, "dropped": True, "hits": dict(hits), "share": cut.get("rows", {}),
            "cloud": cut.get("cloud", 0), "preserved": True, "sweep": False,
            "disk": False, "mint": False}
