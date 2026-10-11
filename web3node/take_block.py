"""Receipt becomes a validated trit block. Cold gets none."""
from take_receipt import receipt
from validate import make, valid
def block(actor, hits, cuts, cold):
    row = receipt(actor, hits, cuts, cold)
    if not row["ok"]:
        return {"ok": False, "why": row["why"], "hash": None}
    check = valid(make(row["hash"]))
    return {"ok": check["ok"], "hash": check.get("hash"), "yield": row["yield"]}
