"""Rate comes from the trit pack of an agreed poll."""
from peers import majority
from validate import make, pack, valid
def rate_of(room, yes):
    votes = [{"who": n, "yes": n in yes} for n in room]
    row = majority(votes, room)
    if not row["agreed"]:
        return {"ok": False, "why": "no_majority", "rate": None}
    body = pack(",".join(sorted(yes)))
    ones = sum(1 for b in body if b == 2)
    rate = round(min(0.2, max(0.01, ones / len(body))), 4)
    check = valid(make(",".join(sorted(yes))))
    return {"ok": check["ok"], "rate": rate, "hash": check.get("hash"),
            "caller_set": False, "constant": False, "socket": False}
