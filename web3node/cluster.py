"""Layer cluster. Membership is a label. This host is not in it."""
LAYERS = {"deck": ["rp2040"], "t4": ["pi5"], "t0": ["m4-mini"], "t1": ["spark"]}
def cluster():
    return {"layers": LAYERS, "members": 0, "present": False,
            "claim_100b_24gb": False, "model_pull": False}
