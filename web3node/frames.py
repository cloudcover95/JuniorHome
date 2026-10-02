"""Activity points for Home frames. No write."""
MODULES = ("gaia", "stonefield", "home", "junioros", "koirig")
def point(name):
    if name not in MODULES:
        return {"ok": False, "why": "unknown", "model_pull": False}
    e = round(sum(ord(c) % 3 for c in name) / len(name), 3)
    return {"ok": True, "name": name, "x": round(e / 3, 3), "y": round(1 - e / 3, 3),
            "energy": e, "protocol": "goldend-osai-omega/1", "write": False, "model_pull": False}
def web():
    return [point(n) for n in MODULES]
