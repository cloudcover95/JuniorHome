"""Full envelope. Real fields only."""
import hashlib
from anticon import fold
from cloud_share import share
from finish import finish
from optin import run
def envelope(user, hits, host, opted_in, role="user"):
    if not run(opted_in)["ok"]:
        return {"ok": False, "why": "isolated"}
    names = [k for k, n in hits.items() for _ in range(n)]
    anti = fold(user, [role], names)
    if not anti["ok"]:
        return {"ok": False, "why": anti["why"]}
    row = finish(hits)
    n = row["hits"]
    cut = share({"actor": n, "host": n if host else 0, "home": n})
    return {"ok": True, "env": "JuniorHome", "role": role, "user": user,
            "session": hashlib.sha256(user.encode()).hexdigest()[:16], "hits": n,
            "kinds": sorted(hits), "hash": row["hash"], "juniors": cut["cloud"],
            "host_share": cut["rows"].get("host", 0), "actor_share": cut["rows"].get("actor", 0),
            "cloud_rate": 0.05, "anticon": True, "packed_once": True, "trit": True,
            "cold": True, "preserved": True, "scheme": "ML-DSA-65", "signed_here": False,
            "gas": False, "mint": False, "pow": False, "socket": False}
