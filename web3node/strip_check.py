"""Fail if a module page lost its status lines. Does not delete."""
NEED = {
    "ui/gaia.html": ("no download",),
    "ui/deck.html": ("flashed false", "MX 0-15"),
    "ui/field.html": ("scrape false",),
    "ui/koirig.html": ("denied",),
}
def check(path, text):
    need = NEED.get(path)
    if need is None:
        return {"ok": True, "path": path, "why": "skip"}
    missing = [n for n in need if n not in text]
    return {"ok": not missing, "path": path, "missing": missing, "delete": False}
