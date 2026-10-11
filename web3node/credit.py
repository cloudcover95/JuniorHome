"""Close credits the Cloud cut to the address. No send."""
from finish import finish
from wallet import wallet
def credit(user, hits):
    row = finish(hits)
    w = wallet(user, row["hits"])
    return {"address": w["address"], "juniors": w["juniors"], "hash": row["hash"],
            "cold": True, "sent": False, "mint": False, "network": False}
