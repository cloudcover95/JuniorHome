"""STE tape on a tiny vector."""
from bitnet_orig import absmean

def step(w, x, target, lr=0.05):
    wq, g = absmean(w)
    y = sum(a * b for a, b in zip(x, [q * g for q in wq]))
    err = y - target
    w2 = [wi - lr * err * xi for wi, xi in zip(w, x)]
    return {"y": y, "err": err, "gamma": g, "w": w2, "ste": True}

if __name__ == "__main__":
    import json
    print(json.dumps({k: step([0.3, -0.2, 0.1], [1.0, 0.5, -0.2], 0.0)[k] for k in ("y", "err", "ste")}))
