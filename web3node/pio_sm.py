"""PIO-shaped scan. RAM only. This box has no RP2040."""
import time
LAYERS = {"t4": {"sm": 1, "steps": 16}, "t0": {"sm": 2, "steps": 64}}
def scan(n):
    return [1 if i % 4 == 0 else 0 for i in range(n)]
def run(layer="t4"):
    spec = LAYERS.get(layer, LAYERS["t4"])
    t0 = time.perf_counter()
    trits = scan(spec["steps"])
    zeros = trits.count(0)
    return {"layer": layer if layer in LAYERS else "t4", "sm": spec["sm"],
            "steps": spec["steps"], "zeros": zeros,
            "energy": round(1.0 - zeros / len(trits), 3),
            "us": round((time.perf_counter() - t0) * 1e6, 1),
            "pio_silicon": False, "disk": False, "model_pull": False}
