"""Portal hits become a trit block. No scan."""
import hashlib
from portal import profile, portal
def pack(text):
    raw = text.encode()[:32]
    xs = [(b - 128) / 128.0 for b in raw]
    g = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / g)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return bytes(t + 1 for t in out)
def block():
    row = portal([profile("home", "dev", {"gaia": 2, "home": 1})])
    node = row["nodes"][0]
    body = pack(str(node["hits"]))
    return {"portal": "JuniorOS", "hash": hashlib.sha256(body).hexdigest()[:16],
            "trit": len(body), "hits": node["hits"], "scan": False, "mint": False,
            "network": False}
