"""Hash failsafe. Lattice is not required to accept the chain."""
from offline import offline
from prove import prove
def failsafe(days=90):
    row = offline(days)
    check = prove(days, row["tip"])
    return {"ok": check["ok"], "backend": "hash", "tip": check["tip"],
            "lattice": False, "slh_dsa": False, "authority": False, "socket": False}
