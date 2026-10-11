"""Close once, then validate the pack."""
from close import close
from validate import make, valid
def finish(hits):
    row = close(hits)
    block = make(str(sorted(hits.items())))
    check = valid(block)
    return {"hits": row["hits"], "hash": check.get("hash"), "ok": check["ok"],
            "packed_once": True, "network": False}
