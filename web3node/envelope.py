"""Block envelope. Every field we actually have."""
from cloud_share import share
from finish import finish
from optin import run
def envelope(user, hits, host, opted_in):
    gate = run(opted_in)
    if not gate["ok"]:
        return {"ok": False, "why": "isolated", "fields": []}
    row = finish(hits)
    n = row["hits"]
    cut = share({"actor": n, "host": n if host else 0, "home": n})
    return {"ok": True, "env": "JuniorHome", "user": user, "hits": n, "hash": row["hash"],
            "juniors": cut["cloud"], "host_share": cut["rows"].get("host", 0),
            "packed_once": True, "trit": True, "gas": False, "mint": False,
            "socket": False, "fields": ["hits", "hash", "juniors", "host_share", "optin"]}
