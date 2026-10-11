"""Node closes once. JuniorOS profiles it. No peer agreement."""
from finish import finish
from portal import profile
def node_close(name, user, hits):
    row = finish(hits)
    card = profile(name, user, hits)
    return {"node": name, "portal": "JuniorOS", "hits": row["hits"], "hash": row["hash"],
            "packed_once": True, "profile": card["hits"], "peers": 0,
            "consensus": False, "network": False}
