"""Discovery is the list you pass. Route only if a node hosts it."""
def discover(nodes):
    live = [n for n in nodes if n.get("host")]
    return {"found": len(live), "peers": len(live), "bind": "127.0.0.1", "scan": False}
def route(content, nodes):
    for n in nodes:
        if content in n.get("hosts", []):
            return {"ok": True, "to": n["name"], "content": content, "gas": False}
    return {"ok": False, "why": "no_host", "content": content, "gas": False}
