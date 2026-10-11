"""First hit in JuniorHome. Not a PiThon handshake."""
from hit import hit
def onboard(name="dev"):
    nodes = [{"name": "home", "host": True, "hosts": ["share"]}]
    row = hit("share", nodes)
    return {"env": "JuniorHome", "dev": name, "ok": row["ok"], "receipt": row.get("receipt"),
            "cloud": row.get("cloud"), "mint": False, "gas": False, "network": False,
            "next": "python3 scripts/hit_block_prod.py"}
