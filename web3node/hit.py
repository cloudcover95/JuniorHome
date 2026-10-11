"""One hosted hit becomes a share and a local receipt. No scan."""
import hashlib
from cloud_share import share
from discover import route
def hit(content, nodes):
    row = route(content, nodes)
    if not row["ok"]:
        return {"ok": False, "why": row["why"], "mint": False, "network": False}
    cut = share({"player": 1, "node": 1})
    body = f"{row['to']}|{content}".encode()
    return {"ok": True, "to": row["to"], "share": cut["rows"], "cloud": cut["cloud"],
            "receipt": hashlib.sha256(body).hexdigest()[:16], "mint": False, "gas": False,
            "network": False}
