"""T4 floor kernel. No numpy."""
from bitnet_orig import absmean
from fleet import isolate
from tnn_layer import bitlinear

def tick(x, w, watts=3.0):
    y = bitlinear(x, w)["y"]
    q, g = absmean(x)
    return {"y": y, "gamma": g, "n": len(q), "node": isolate({"workload": "telemetry"}, watts), "numpy": False}

if __name__ == "__main__":
    print(tick([0.2, -0.1, 0.4], [0.3, 0.0, -0.2]))
