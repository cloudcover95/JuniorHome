"""Two different sessions. Second names the first hash."""
from envelope_full import envelope
from validate import make, valid
def one(user, hits, prev):
    row = envelope(user, hits, True, True, "node")
    if not row["ok"]:
        return {"ok": False}
    block = make(row["hash"], prev)
    check = valid(block)
    return {"ok": check["ok"], "hash": block["hash"], "prev": prev}
def chain():
    a = one("dev", {"prompt": 4}, "0" * 16)
    b = one("home", {"gaia": 2}, a["hash"])
    return {"ok": a["ok"] and b["ok"] and b["prev"] == a["hash"],
            "first": a["hash"], "second": b["hash"], "network": False}
