"""Isolated unless opted in. Default does not touch Home."""
def run(opted_in, kind="session"):
    if not opted_in:
        return {"ok": False, "why": "isolated", "touched": False, "env": None, "kind": kind}
    return {"ok": True, "why": "opted_in", "touched": True, "env": "JuniorHome", "kind": kind}
