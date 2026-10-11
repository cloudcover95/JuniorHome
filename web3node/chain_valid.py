"""Two blocks link only if both are valid trits."""
from validate import make, valid
def chain(a, b):
    first = make(a)
    second = make(b, first["hash"])
    ok = valid(first)["ok"] and valid(second)["ok"] and second["prev"] == first["hash"]
    return {"ok": ok, "first": first["hash"], "second": second["hash"],
            "linked": second["prev"] == first["hash"], "network": False}
