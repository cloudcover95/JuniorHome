"""JuniorOS discovery portal. Profiles are passed in. No scan."""
from node_host import STACK
def profile(name, user, hits):
    return {"name": name, "user": user, "stack": list(STACK),
            "hits": {k: hits.get(k, 0) for k in STACK}, "portal": "JuniorOS",
            "scan": False, "network": False}
def portal(nodes):
    return {"portal": "JuniorOS", "n": len(nodes), "nodes": nodes, "scan": False}
