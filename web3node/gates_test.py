"""Deny paths. A missing device is not a load."""
import sys
from load import load
from route import route
def check():
    no_device = load("/no/such/model.gguf", False)
    missing = load("/no/such/model.gguf", True)
    unknown = route("nope", {"pi": True})
    ok = (no_device["why"] == "no_device" and missing["why"] == "missing"
          and unknown["why"] == "unknown" and no_device["model_pull"] is False)
    return {"ok": ok, "no_device": no_device["why"], "missing": missing["why"], "unknown": unknown["why"]}
if __name__ == "__main__":
    import json
    row = check()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
