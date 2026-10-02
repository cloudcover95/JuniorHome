"""Adjacent nodes. A 100B-in-24GB claim is not a measurement here."""
NODES = ({"name": "deck", "class": "rp2040", "job": "controls", "gb": 0},
         {"name": "t4", "class": "pi5", "job": "trit ticket", "gb": 8},
         {"name": "t0", "class": "m4-mini", "job": "mlx if present", "gb": 24},
         {"name": "t1", "class": "spark", "job": "ue only if latched", "gb": 128})
def stack():
    return {"nodes": list(NODES), "claim_100b_24gb": False, "measured": False,
            "admit": False, "model_pull": False}
