"""Gated close with the cut in the pack."""
from osai_gate import gate
from reward import reward
from validate import make, valid
def close(actor, hits, prev="0" * 16):
    row = gate(actor, hits, True)
    if not row["ok"]:
        return {"ok": False, "why": row["why"]}
    pay = reward(len(hits), True)
    body = f"{row['hash']}|{pay['rate']}|{pay['juniors']}"
    block = make(body, prev)
    check = valid(block)
    return {"ok": check["ok"], "hash": block["hash"], "prev": prev,
            "rate": pay["rate"], "juniors": pay["juniors"], "in_pack": True, "mint": False}
