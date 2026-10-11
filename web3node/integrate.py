"""One path. Opt in, AntiCon, envelope, trit check, link."""
import sys
from anticon import fold
from envelope_full import envelope
from optin import run
from validate import make, valid
def integrate():
    if not run(True)["ok"] or run(False)["ok"]:
        return {"ok": False, "why": "optin"}
    if fold("", ["user"], ["prompt"])["ok"]:
        return {"ok": False, "why": "bot"}
    a = envelope("dev", {"prompt": 4}, True, True, "node")
    b = envelope("home", {"gaia": 2}, True, True, "user")
    if not (a["ok"] and b["ok"]):
        return {"ok": False, "why": "envelope"}
    ba = make(a["hash"])
    bb = make(b["hash"], ba["hash"])
    if not (valid(ba)["ok"] and valid(bb)["ok"] and bb["prev"] == ba["hash"]):
        return {"ok": False, "why": "link"}
    return {"ok": True, "first": ba["hash"], "second": bb["hash"], "mint": False, "socket": False}
if __name__ == "__main__":
    import json
    row = integrate()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
