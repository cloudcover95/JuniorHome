"""Koirig. Land is controls. Sea is automation. Goldend stays the protocol."""
PROTOCOL = "goldend-osai-omega/1"
LAND = ("note", "step", "gamma", "filter", "clock", "mix")
SEA = ("tick", "gate", "envelope")
def route(kind, name):
    if kind == "land" and name in LAND:
        return {"ok": True, "engine": "koirig", "side": "land", "name": name,
                "protocol": PROTOCOL, "write": False, "model_pull": False}
    if kind == "sea" and name in SEA:
        return {"ok": True, "engine": "koirig", "side": "sea", "name": name,
                "protocol": PROTOCOL, "write": False, "model_pull": False}
    return {"ok": False, "engine": "koirig", "why": "crossed", "model_pull": False}
