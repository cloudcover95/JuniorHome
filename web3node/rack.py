"""Planned rack. Slots are not attached on this host."""
SLOTS = ({"class": "pi5", "n": 4, "job": "trit ticket"},
         {"class": "gpu", "n": 5, "job": "admit if nvidia-smi"},
         {"class": "spark", "n": 1, "job": "ue only if latched"},
         {"class": "m5-pro", "n": 1, "job": "mlx if present"})
def rack():
    return {"slots": list(SLOTS), "planned": sum(s["n"] for s in SLOTS), "attached": 0,
            "claim_100b_24gb": False, "model_pull": False}
