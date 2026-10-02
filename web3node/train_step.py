"""One training step. Missing file or device means zero steps."""
from load import load
def step(path, present):
    row = load(path, present)
    if not row["ok"]:
        return {"steps": 0, "trained": False, "why": row["why"], "model_pull": False}
    return {"steps": 1, "trained": False, "why": "local", "model_pull": False}
