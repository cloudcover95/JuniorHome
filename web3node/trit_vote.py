"""A vote is a trit pack. Counted once. Local."""
from peers import majority, peers
from validate import make, valid
def trit_vote(room, yes):
    names = peers(room)["peers"]
    votes = [{"who": n, "yes": n in yes} for n in names]
    row = majority(votes, names)
    check = valid(make(",".join(names)))
    return {"ok": row["agreed"] and check["ok"], "agreed": row["agreed"],
            "hash": check.get("hash"), "trit": True, "packed_once": True,
            "env": "JuniorHome", "socket": False}
