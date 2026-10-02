"""Second brain. Six ports. Gaia asks. OSai routes. No model pull."""
PORTS = ("JuniorLLM", "AGI_SDK", "JuniorOSai", "JuniorOS", "web3node", "obsidian")
def note(text, who="gaia"):
    raw = (text or "x")[:64]
    gamma = sum(ord(c) % 3 for c in raw) / len(raw)
    return {"who": who, "ports": list(PORTS), "n": len(raw), "gamma": round(gamma, 3),
            "vault": "obsidian", "synced": False, "admit": False,
            "ram_engine": False, "model_pull": False}
