"""One stack tick. Gaia asks. Deck controls. No vault write."""
PORTS = ("JuniorLLM", "AGI_SDK", "JuniorOSai", "JuniorOS", "web3node", "obsidian")
def tick():
    return {"gaia": {"role": "helper", "ask": "note"},
            "deck": {"role": "controls", "control": "gamma", "pin": "ADC 26"},
            "ports": list(PORTS), "synced": False, "admit": False, "disk": False,
            "opened": False, "model_pull": False}
