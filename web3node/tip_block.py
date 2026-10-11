"""Hash tip is the parent of the next trit block."""
from anticon_hash import guard
from validate import make, valid
def next_block(actor="dev"):
    row = guard(actor, ["gaia", "home"])
    if not row["ok"]:
        return {"ok": False, "why": row["why"]}
    block = make("next", row["tip"])
    check = valid(block)
    return {"ok": check["ok"] and block["prev"] == row["tip"], "prev": block["prev"],
            "hash": block["hash"], "backend": "hash"}
