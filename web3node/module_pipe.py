"""Module pipeline. Separate identities. No weight clone."""
MODULES = {
    "gaia": {"role": "helper", "pins": False},
    "deck": {"role": "controls", "pins": True},
}
def create(name):
    row = MODULES.get(name)
    if row is None:
        return {"ok": False, "why": "unknown", "model_pull": False}
    return {"ok": True, "name": name, "role": row["role"], "pins": row["pins"],
            "clone_of": None, "download_gb": 0.0, "model_pull": False}
def route(name):
    body = create(name)
    body["agent"] = "osai" if body.get("ok") else "deny"
    return body
