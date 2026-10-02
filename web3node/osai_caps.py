"""OSai capabilities by envelope. Gaia and Deck stay separate."""
CAPS = {"gaia": {"t4": ["ask", "label"], "t0": ["ask", "label", "note"]},
        "deck": {"t4": ["step", "note"], "t0": ["step", "note", "mix"]}}
def expand(name, layer, present):
    verbs = CAPS.get(name, {}).get(layer, [])
    return {"name": name, "layer": layer, "verbs": verbs if present else [],
            "admit": bool(present and verbs), "clone_of": None,
            "ram_engine": False, "model_pull": False}
