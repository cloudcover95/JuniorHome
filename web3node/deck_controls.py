"""JuniorDeck controls on the production package. Not Gaia."""
PORTS = ("JuniorLLM", "AGI_SDK", "JuniorOSai", "JuniorOS", "web3node", "obsidian")
CONTROLS = {"note": "MX 0-15", "step": "pads 16-23", "gamma": "ADC 26",
            "filter": "ADC 27", "clock": "ADC 28", "mix": "ADC 29", "surface": "DSI"}
def control(name="note"):
    if name not in CONTROLS:
        return {"ok": False, "why": "unknown", "model_pull": False}
    return {"ok": True, "who": "deck", "control": name, "pin": CONTROLS[name],
            "ports": list(PORTS), "clone_of": None, "synced": False, "admit": False,
            "opened": False, "ram_engine": False, "model_pull": False}
