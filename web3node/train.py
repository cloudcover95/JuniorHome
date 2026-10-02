"""Training program. Names the three. Does not train on this host."""
PORTS = ("JuniorLLM", "AGI_SDK", "Gaia")
def program():
    return {"ports": list(PORTS), "steps": 0, "trained": False, "device": False, "model_pull": False}
